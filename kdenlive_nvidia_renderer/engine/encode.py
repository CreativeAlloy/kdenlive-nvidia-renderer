"""NVENC output: raw NV12 frames in on stdin, encoded file out."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from fractions import Fraction

from .media import AUDIO_CHANNELS, AUDIO_RATE, FFMPEG

CODECS = {
    "H.264 (NVENC)": "h264_nvenc",
    "HEVC / H.265 (NVENC)": "hevc_nvenc",
    "AV1 (NVENC)": "av1_nvenc",
}

PRESETS = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]


@dataclass
class EncodeSettings:
    output: str
    width: int
    height: int
    fps: Fraction
    codec: str = "h264_nvenc"
    preset: str = "p5"
    rate_control: str = "cq"          # "cq" (constant quality) or "vbr" (target bitrate)
    quality: int = 19                  # CQ value, lower = better
    bitrate_mbps: float = 20.0
    audio: bool = True
    audio_codec: str = "aac"
    audio_bitrate_k: int = 320
    colorspace: int = 709
    lossless_debug: bool = False       # RGB in -> FFV1 4:4:4 out (fidelity testing)


_CS_TAGS = {709: ("bt709", "bt709", "bt709"), 601: ("smpte170m", "smpte170m", "smpte170m"),
            2020: ("bt2020nc", "bt2020", "bt709")}


def build_command(s: EncodeSettings, audio_path: str | None) -> list[str]:
    if s.lossless_debug:
        cs = "bt709" if s.colorspace == 709 else "smpte170m"
        return [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-f", "rawvideo", "-pix_fmt", "yuv444p", "-s", f"{s.width}x{s.height}",
                "-framerate", f"{s.fps.numerator}/{s.fps.denominator}", "-i", "pipe:0",
                "-c:v", "ffv1", "-colorspace", cs, s.output]
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-f", "rawvideo", "-pix_fmt", "nv12", "-s", f"{s.width}x{s.height}",
           "-framerate", f"{s.fps.numerator}/{s.fps.denominator}", "-i", "pipe:0"]
    if s.audio and audio_path:
        cmd += ["-f", "f32le", "-ar", str(AUDIO_RATE), "-ac", str(AUDIO_CHANNELS), "-i", audio_path]
    cmd += ["-map", "0:v"]
    if s.audio and audio_path:
        cmd += ["-map", "1:a"]
    cmd += ["-c:v", s.codec, "-preset", s.preset, "-tune", "hq"]
    if s.rate_control == "cq":
        cmd += ["-rc", "vbr", "-cq", str(s.quality), "-b:v", "0"]
    else:
        br = f"{s.bitrate_mbps:g}M"
        cmd += ["-rc", "vbr", "-b:v", br, "-maxrate", f"{s.bitrate_mbps * 1.5:g}M",
                "-bufsize", f"{s.bitrate_mbps * 2:g}M"]
    cmd += ["-spatial-aq", "1", "-temporal-aq", "1", "-rc-lookahead", "32"]
    if s.codec == "h264_nvenc":
        cmd += ["-profile:v", "high", "-bf", "3", "-b_ref_mode", "middle"]
    elif s.codec == "hevc_nvenc":
        cmd += ["-profile:v", "main", "-bf", "3", "-b_ref_mode", "middle", "-tag:v", "hvc1"]
    cs, prim, trc = _CS_TAGS.get(s.colorspace, _CS_TAGS[709])
    cmd += ["-colorspace", cs, "-color_primaries", prim, "-color_trc", trc, "-color_range", "tv"]
    if s.audio and audio_path:
        if s.audio_codec == "flac":
            cmd += ["-c:a", "flac"]
        elif s.audio_codec == "pcm":
            cmd += ["-c:a", "pcm_s16le"]
        else:
            cmd += ["-c:a", "aac", "-b:a", f"{s.audio_bitrate_k}k"]
    ext = os.path.splitext(s.output)[1].lower()
    if ext in (".mp4", ".mov", ".m4v"):
        cmd += ["-movflags", "+faststart"]
    cmd += [s.output]
    return cmd


class Encoder:
    """Feeds NV12 frames to FFmpeg/NVENC from a background thread."""

    def __init__(self, settings: EncodeSettings, audio_path: str | None):
        self.settings = settings
        self.cmd = build_command(settings, audio_path)
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=settings.width * settings.height * 3)
        self._q: queue.Queue = queue.Queue(maxsize=6)
        self._error: BaseException | None = None
        self._stderr = b""
        self._err_thread = threading.Thread(target=self._read_err, daemon=True)
        self._err_thread.start()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def _read_err(self):
        try:
            self._stderr = self.proc.stderr.read() or b""
        except Exception:
            pass

    def _writer(self):
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                self.proc.stdin.write(item)
        except BaseException as e:   # broken pipe etc.
            self._error = e
            # drain so producers never block
            while True:
                try:
                    if self._q.get(timeout=0.1) is None:
                        break
                except queue.Empty:
                    if self.proc.poll() is not None:
                        break

    def write(self, frame_bytes):
        if self._error is not None:
            raise RuntimeError(self.error_text())
        self._q.put(frame_bytes)

    def error_text(self) -> str:
        self._err_thread.join(timeout=2)
        return "FFmpeg encoder failed:\n" + self._stderr.decode(errors="replace")[-1500:]

    def finish(self) -> None:
        self._q.put(None)
        self._thread.join()
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        code = self.proc.wait()
        self._err_thread.join(timeout=5)
        if code != 0 or self._error is not None:
            raise RuntimeError(self.error_text())

    def abort(self):
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
