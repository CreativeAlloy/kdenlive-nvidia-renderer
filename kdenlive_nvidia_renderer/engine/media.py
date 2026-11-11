"""Media probing and streaming FFmpeg readers.

Video is decoded by NVDEC (``-hwaccel cuda``) and, when the source is larger
than needed, downscaled on the GPU by ``scale_cuda`` before being handed over
as raw NV12/P010.  Codecs NVDEC cannot handle fall back to software decoding
automatically.  Audio is decoded to 48 kHz stereo float32.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np

FFMPEG = os.environ.get("KNR_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = os.environ.get("KNR_FFPROBE") or shutil.which("ffprobe") or "ffprobe"

NVDEC_CODECS = {"h264", "hevc", "av1", "vp9", "vp8", "mpeg2video", "mpeg1video", "mpeg4", "vc1",
                "mjpeg"}
IMAGE_CODECS = {"png", "mjpeg", "bmp", "tiff", "webp", "gif", "jpegls", "jpeg2000", "pam", "ppm",
                "pgm", "svg", "exr", "qoi"}

AUDIO_RATE = 48000
AUDIO_CHANNELS = 2


def _parse_ratio(s: str | None, default=Fraction(0)) -> Fraction:
    if not s or s in ("0/0", "N/A"):
        return default
    try:
        if ":" in s:
            a, b = s.split(":")
            return Fraction(int(a), int(b)) if int(b) else default
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return default


@dataclass
class VideoInfo:
    index: int
    codec: str
    width: int
    height: int
    pix_fmt: str
    fps: Fraction
    sar: Fraction
    rotation: int
    colorspace: int
    full_range: bool
    bit_depth: int
    nb_frames: int
    duration: float
    is_image: bool


@dataclass
class AudioInfo:
    index: int
    codec: str
    sample_rate: int
    channels: int


@dataclass
class MediaInfo:
    path: str
    duration: float
    video: list[VideoInfo]
    audio: list[AudioInfo]

    def video_stream(self, index: int | None) -> VideoInfo | None:
        if index is not None and index >= 0:
            for v in self.video:
                if v.index == index:
                    return v
        return self.video[0] if self.video else None

    def audio_stream(self, index: int | None) -> AudioInfo | None:
        if index is not None and index >= 0:
            for a in self.audio:
                if a.index == index:
                    return a
        return self.audio[0] if self.audio else None


_CS_MAP = {"bt709": 709, "bt470bg": 601, "smpte170m": 601, "bt2020nc": 2020, "bt2020c": 2020,
           "smpte240m": 240, "fcc": 601}


@lru_cache(maxsize=256)
def probe(path: str) -> MediaInfo:
    cmd = [FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {res.stderr.strip()[:500]}")
    data = json.loads(res.stdout or "{}")
    fmt = data.get("format", {})
    fmt_name = fmt.get("format_name", "")
    duration = float(fmt.get("duration") or 0)
    videos, audios = [], []
    for st in data.get("streams", []):
        kind = st.get("codec_type")
        if kind == "video":
            if (st.get("disposition") or {}).get("attached_pic"):
                continue
            rot = 0
            for sd in st.get("side_data_list", []) or []:
                if "rotation" in sd:
                    rot = int(round(float(sd["rotation"])))
            if "rotate" in (st.get("tags") or {}):
                rot = -int(st["tags"]["rotate"])
            rot = (-rot) % 360          # clockwise degrees needed to display upright
            fps = _parse_ratio(st.get("r_frame_rate")) or _parse_ratio(st.get("avg_frame_rate"))
            h = int(st.get("height") or 0)
            cs = _CS_MAP.get(st.get("color_space") or "", 0) or (601 if h and h < 720 else 709)
            pix = st.get("pix_fmt") or ""
            depth = int(st.get("bits_per_raw_sample") or 0) or (
                10 if any(t in pix for t in ("10", "12", "16")) else 8)
            codec = st.get("codec_name") or ""
            is_image = codec in IMAGE_CODECS and (
                "image2" in fmt_name or "_pipe" in fmt_name or int(st.get("nb_frames") or 1) <= 1)
            videos.append(VideoInfo(
                index=int(st["index"]), codec=codec, width=int(st.get("width") or 0), height=h,
                pix_fmt=pix, fps=fps or Fraction(25), sar=_parse_ratio(st.get("sample_aspect_ratio"), Fraction(1)) or Fraction(1),
                rotation=rot, colorspace=cs,
                full_range=(st.get("color_range") == "pc") or pix.startswith("yuvj"),
                bit_depth=depth, nb_frames=int(st.get("nb_frames") or 0),
                duration=float(st.get("duration") or duration or 0), is_image=is_image))
        elif kind == "audio":
            audios.append(AudioInfo(index=int(st["index"]), codec=st.get("codec_name") or "",
                                    sample_rate=int(st.get("sample_rate") or 48000),
                                    channels=int(st.get("channels") or 2)))
    return MediaInfo(path=path, duration=duration, video=videos, audio=audios)


# ---------------------------------------------------------------------------- video

@dataclass(frozen=True)
class VideoRequest:
    """Everything needed to (re)start a decoder process."""
    path: str
    stream: int
    out_w: int            # decoded frame size delivered to us (pre-rotation)
    out_h: int
    fps: Fraction         # timeline frame rate
    speed: float = 1.0
    same_rate: bool = True


class DecodeError(RuntimeError):
    pass


class VideoReader:
    """Streams consecutive frames of one clip instance from an FFmpeg process.

    Frames are indexed in *timeline* frame units relative to the source start,
    i.e. index ``i`` is the picture shown at source time ``i / fps * speed``.
    """

    PREFETCH = 8

    def __init__(self, req: VideoRequest, info: VideoInfo, start: int, use_hw: bool = True,
                 log=None):
        self.req = req
        self.info = info
        self.is16 = info.bit_depth > 8
        self.frame_bytes = req.out_w * req.out_h * 3 // 2 * (2 if self.is16 else 1)
        self.log = log or (lambda *_: None)
        self.use_hw = use_hw and info.codec in NVDEC_CODECS and not info.is_image
        self._proc = None
        self._thread = None
        self._queue: queue.Queue = queue.Queue(maxsize=self.PREFETCH)
        self._stop = threading.Event()
        self.next_index = start
        self.last: np.ndarray | None = None
        self.eof = False
        self._start(start)

    # -- process management
    def _cmd(self, start: int, hw: bool) -> list[str]:
        r = self.req
        t = float(start) / float(r.fps) * r.speed
        fmt = "p010le" if self.is16 else "nv12"
        filters = []
        if r.same_rate and r.speed == 1.0:
            # Seek half a frame early so the frame exactly at ``t`` is the first one kept.
            t = max(0.0, t - 0.5 / float(r.fps))
        else:
            if r.speed != 1.0:
                filters.append(f"setpts=(PTS-STARTPTS)/{r.speed:.10g}")
            filters.append(f"fps={r.fps.numerator}/{r.fps.denominator}:round=near")
        cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-noautorotate"]
        need_scale = (r.out_w, r.out_h) != (self.info.width, self.info.height)
        if hw:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            if need_scale:
                filters.append(f"scale_cuda={r.out_w}:{r.out_h}:interp_algo=lanczos:format={fmt}")
            else:
                filters.append(f"scale_cuda=format={fmt}")
            filters.append(f"hwdownload,format={fmt}")
        else:
            if need_scale:
                filters.append(f"scale={r.out_w}:{r.out_h}:flags=lanczos")
            filters.append(f"format={fmt}")
        if t > 0:
            cmd += ["-ss", f"{t:.6f}"]
        cmd += ["-i", r.path, "-map", f"0:{r.stream}", "-an", "-sn", "-dn",
                "-vf", ",".join(filters), "-fps_mode", "passthrough",
                "-f", "rawvideo", "-pix_fmt", fmt, "pipe:1"]
        return cmd

    def _start(self, start: int):
        self.close()
        self._stop = threading.Event()
        self._queue = queue.Queue(maxsize=self.PREFETCH)
        self.next_index = start
        self.eof = False
        self._produced = 0
        cmd = self._cmd(start, self.use_hw)
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      bufsize=self.frame_bytes * 2)
        self._thread = threading.Thread(target=self._pump, args=(self._proc, self._queue, self._stop),
                                        daemon=True)
        self._thread.start()

    def _pump(self, proc, q, stop):
        fb = self.frame_bytes
        stdout = proc.stdout
        try:
            while not stop.is_set():
                buf = bytearray(fb)
                view = memoryview(buf)
                got = 0
                while got < fb:
                    n = stdout.readinto(view[got:])
                    if not n:
                        break
                    got += n
                if got < fb:
                    break
                while not stop.is_set():
                    try:
                        q.put(np.frombuffer(buf, dtype=np.uint8), timeout=0.2)
                        break
                    except queue.Full:
                        continue
        finally:
            err = b""
            try:
                err = proc.stderr.read() or b""
            except Exception:
                pass
            if not stop.is_set():
                while True:
                    try:
                        q.put(("EOF", proc.wait(), err.decode(errors="replace")), timeout=0.2)
                        break
                    except queue.Full:
                        if stop.is_set():
                            break

    def _next(self) -> np.ndarray | None:
        item = self._queue.get()
        if isinstance(item, tuple):
            _, code, err = item
            if self._produced == 0 and self.use_hw:
                # Hardware path failed before producing anything: retry in software.
                tail = err.strip().splitlines()[-1] if err.strip() else ""
                self.log(f"NVDEC unavailable for {os.path.basename(self.req.path)}; "
                         f"falling back to software decoding. {tail}".strip())
                self.use_hw = False
                self._start(self.next_index)
                return self._next()
            if code not in (0, None) and self._produced == 0 and self.last is None:
                raise DecodeError(f"FFmpeg could not decode {self.req.path}:\n{err.strip()[-800:]}")
            self.eof = True
            return None
        self._produced += 1
        self.next_index += 1
        return item

    def read(self, index: int) -> np.ndarray | None:
        """Return the frame at ``index`` (sequential access is cheap, seeking restarts FFmpeg)."""
        if self.last is not None and index == self.next_index - 1:
            return self.last
        max_skip = 2 * max(1, int(float(self.req.fps)))
        if index < self.next_index or index > self.next_index + max_skip:
            self._start(index)
            self.last = None
        while not self.eof and self.next_index <= index:
            frame = self._next()
            if frame is None:
                break
            self.last = frame
        return self.last

    def close(self):
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            try:
                self._proc.stdout.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=2)
            except Exception:
                pass
        self._proc = None


def load_image_rgba(path: str, frame_time: float = 0.0) -> tuple[np.ndarray, int, int]:
    """Decode a still image (or one frame of a video) to an RGBA8 buffer."""
    info = probe(path)
    v = info.video_stream(None)
    if v is None:
        raise DecodeError(f"No picture in {path}")
    w, h = v.width, v.height
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error"]
    if frame_time > 0:
        cmd += ["-ss", f"{frame_time:.6f}"]
    cmd += ["-i", path, "-map", f"0:{v.index}", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or len(res.stdout) < w * h * 4:
        raise DecodeError(f"Could not decode image {path}: {res.stderr.decode(errors='replace')[-500:]}")
    return np.frombuffer(res.stdout[: w * h * 4], dtype=np.uint8), w, h


# ---------------------------------------------------------------------------- audio

class AudioReader:
    """Streams 48 kHz stereo float32 samples for one clip instance."""

    CHUNK = 4096

    def __init__(self, path: str, stream: int, start_sample: int, speed: float = 1.0,
                 pitch_compensate: bool = False):
        self.path = path
        self.stream = stream
        self.speed = speed
        self.pitch = pitch_compensate
        self._proc = None
        self.pos = start_sample
        self._buf = np.zeros((0, AUDIO_CHANNELS), dtype=np.float32)
        self.eof = False
        self._start(start_sample)

    def _start(self, start_sample: int):
        self.close()
        self.pos = start_sample
        self.eof = False
        self._buf = np.zeros((0, AUDIO_CHANNELS), dtype=np.float32)
        t = start_sample / AUDIO_RATE * self.speed
        filters = [f"aresample={AUDIO_RATE}"]
        if self.speed != 1.0:
            if self.pitch:
                sp = self.speed
                chain = []
                while sp > 2.0:
                    chain.append("atempo=2.0")
                    sp /= 2.0
                while sp < 0.5:
                    chain.append("atempo=0.5")
                    sp /= 0.5
                chain.append(f"atempo={sp:.10g}")
                filters = chain + filters
            else:
                filters = [f"aresample={AUDIO_RATE}", f"asetrate={AUDIO_RATE * self.speed:.6f}",
                           f"aresample={AUDIO_RATE}"]
        cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error"]
        if t > 0:
            cmd += ["-ss", f"{t:.6f}"]
        cmd += ["-i", self.path, "-map", f"0:{self.stream}", "-vn", "-sn", "-dn",
                "-af", ",".join(filters), "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_RATE),
                "-f", "f32le", "pipe:1"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self, start_sample: int, n: int) -> np.ndarray:
        """Read ``n`` samples at ``start_sample``.

        Consecutive frames may disagree by a sample or two on where they start
        (timeline vs. source frame rounding); those are treated as contiguous.
        """
        delta = start_sample - self.pos
        if abs(delta) > 2 * max(n, 1):
            if 0 < delta < AUDIO_RATE:
                self.read(self.pos, delta)                        # skip forward
            else:
                self._start(start_sample)
        start_sample = self.pos
        while len(self._buf) < n and not self.eof:
            raw = self._proc.stdout.read(self.CHUNK * AUDIO_CHANNELS * 4)
            if not raw:
                self.eof = True
                break
            usable = len(raw) - len(raw) % (AUDIO_CHANNELS * 4)
            arr = np.frombuffer(raw[:usable], dtype=np.float32).reshape(-1, AUDIO_CHANNELS)
            self._buf = np.concatenate([self._buf, arr]) if len(self._buf) else arr
        out = self._buf[:n]
        self._buf = self._buf[n:]
        if len(out) < n:
            out = np.concatenate([out, np.zeros((n - len(out), AUDIO_CHANNELS), dtype=np.float32)])
        self.pos = start_sample + n
        return out

    def close(self):
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.stdout.close()
                self._proc.wait(timeout=2)
            except Exception:
                pass
        self._proc = None
