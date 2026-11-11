"""Render orchestration: audio pass, then GPU video pass feeding NVENC."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import dataclass
from fractions import Fraction

from ..mlt.project import Project
from .encode import EncodeSettings


@dataclass
class Progress:
    stage: str            # "audio", "video", "finishing", "done"
    done: int
    total: int
    fps: float = 0.0
    eta: float | None = None
    elapsed: float = 0.0


class Cancelled(Exception):
    pass


def render(project: Project, settings: EncodeSettings, frame_range: tuple[int, int] | None = None,
           progress=None, log=None, cancel: threading.Event | None = None) -> float:
    """Render ``project`` to ``settings.output``. Returns elapsed seconds."""
    import cupy as cp

    from . import gpu
    from .audio import AudioExecutor
    from .context import RenderContext
    from .encode import Encoder
    from .timeline import Timeline
    from .video import VideoExecutor

    log = log or (lambda m: None)
    progress = progress or (lambda p: None)
    cancel = cancel or threading.Event()
    t0 = time.monotonic()

    if settings.width % 2 or settings.height % 2:
        raise ValueError("Output width and height must be even numbers")

    tl = Timeline(project)
    start, end = frame_range or tl.range
    profile_fps = project.profile.fps_fraction
    out_fps = settings.fps or profile_fps
    n_timeline = end - start + 1
    n_out = int((Fraction(n_timeline) * out_fps / profile_fps))
    n_out = max(1, n_out)

    ctx = RenderContext(project, settings.width, settings.height, log=log)
    settings.colorspace = ctx.colorspace
    keep = int(float(profile_fps) * 2) + 2

    # ------------------------------------------------------------------ audio pass
    audio_path = None
    if settings.audio:
        fd, audio_path = tempfile.mkstemp(prefix="knr_audio_", suffix=".f32")
        os.close(fd)
        aex = AudioExecutor(ctx)
        try:
            with open(audio_path, "wb") as fh:
                for i, pos in enumerate(range(start, end + 1)):
                    if cancel.is_set():
                        raise Cancelled()
                    node = tl.evaluate(pos)
                    fh.write(aex.render(node, pos).tobytes())
                    if i % 500 == 0:
                        aex.gc(keep)
                        progress(Progress("audio", i, n_timeline, elapsed=time.monotonic() - t0))
        except BaseException:
            aex.close()
            _unlink(audio_path)
            raise
        aex.close()
        log(f"Audio mixed in {time.monotonic() - t0:.1f}s")

    # ------------------------------------------------------------------ video pass
    gpu.warmup()
    vex = VideoExecutor(ctx)
    enc = None
    try:
        enc = Encoder(settings, audio_path)
        log("Encoder: " + " ".join(enc.cmd))
        lookahead = max(2, int(float(profile_fps) * 0.75))
        nv12 = cp.empty(settings.width * settings.height * 3 // 2, dtype=cp.uint8)
        t_video = time.monotonic()
        for k in range(n_out):
            if cancel.is_set():
                raise Cancelled()
            # map output frame -> timeline position (nearest earlier frame when rates differ)
            if out_fps == profile_fps:
                pos = start + k
            else:
                pos = start + int(Fraction(k) * profile_fps / out_fps)
            node = tl.evaluate(pos)
            if pos + lookahead <= end:
                vex.prefetch(tl.evaluate(pos + lookahead))
            canvas = vex.render(node)
            if settings.lossless_debug:
                enc.write(cp.asnumpy(gpu.rgba_to_yuv444(canvas, ctx.colorspace)))
            else:
                gpu.rgba_to_nv12(canvas, ctx.colorspace, out=nv12)
                enc.write(cp.asnumpy(nv12))
            if k % 30 == 0:
                vex.gc(keep)
                el = time.monotonic() - t_video
                fps = (k + 1) / el if el > 0 else 0.0
                eta = (n_out - k - 1) / fps if fps > 0 else None
                progress(Progress("video", k + 1, n_out, fps=fps, eta=eta, elapsed=time.monotonic() - t0))
        progress(Progress("finishing", n_out, n_out, elapsed=time.monotonic() - t0))
        enc.finish()
        enc = None
    except BaseException:
        if enc is not None:
            enc.abort()
        vex.close()
        if audio_path:
            _unlink(audio_path)
        # remove partial output on cancel/failure
        _unlink(settings.output)
        raise
    vex.close()
    if audio_path:
        _unlink(audio_path)
    elapsed = time.monotonic() - t0
    progress(Progress("done", n_out, n_out, elapsed=elapsed))
    return elapsed


def _unlink(path):
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
