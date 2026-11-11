"""Audio executor: mixes the timeline's audio frame by frame (48 kHz stereo float32).

Audio is cheap compared to video, so it is mixed on the CPU with numpy while
FFmpeg decodes the sources.  Frame sample counts follow MLT so audio stays in
sync with video over arbitrarily long timelines.
"""

from __future__ import annotations

import os

import numpy as np

from .context import RenderContext
from .effects import AUDIO_FILTERS, AUDIO_TRANSITIONS, IGNORED, VIDEO_FILTERS
from .media import AUDIO_CHANNELS, AUDIO_RATE, AudioReader, probe
from .timeline import ClipNode, Node, StackNode, TransitionUse

AUDIO_SERVICES = {"avformat", "avformat-novalidate", "timewarp"}


class AudioExecutor:
    def __init__(self, ctx: RenderContext):
        self.ctx = ctx
        self.fps = ctx.fps_fraction
        self.readers: dict[tuple, AudioReader] = {}
        self._last_used: dict[tuple, int] = {}
        self._tick = 0

    def sample_start(self, frame: int) -> int:
        return (frame * AUDIO_RATE * self.fps.denominator) // self.fps.numerator

    def samples_for(self, frame: int) -> int:
        return self.sample_start(frame + 1) - self.sample_start(frame)

    def render(self, node: Node | None, frame: int) -> np.ndarray:
        self._tick += 1
        n = self.samples_for(frame)
        out = self._node(node, n) if node is not None else None
        if out is None:
            return np.zeros((n, AUDIO_CHANNELS), dtype=np.float32)
        return out.astype(np.float32, copy=False)

    def gc(self, keep_frames: int):
        dead = [k for k, t in self._last_used.items() if self._tick - t > keep_frames]
        for k in dead:
            r = self.readers.pop(k, None)
            if r:
                r.close()
            self._last_used.pop(k, None)

    def close(self):
        for r in self.readers.values():
            r.close()
        self.readers.clear()

    # ------------------------------------------------------------------
    def _node(self, node: Node, n: int) -> np.ndarray | None:
        if node.no_audio:
            return None
        if isinstance(node, ClipNode):
            buf = self._clip(node, n)
        elif isinstance(node, StackNode):
            buf = self._stack(node, n)
        else:
            buf = None
        if buf is None:
            return None
        for use in node.filters:
            svc = use.filter.service
            fn = AUDIO_FILTERS.get(svc)
            if fn is None:
                if svc not in IGNORED and svc not in VIDEO_FILTERS:
                    self.ctx.warn(f"Skipping unsupported audio effect '{use.filter.kdenlive_id}' ({svc})")
                continue
            buf = fn(self.ctx, buf, use, None)
        return buf

    def _stack(self, node: StackNode, n: int) -> np.ndarray | None:
        bufs: list[np.ndarray | None] = []
        for i, child in enumerate(node.tracks):
            if child is None or node.hide_audio[i]:
                bufs.append(None)
            else:
                bufs.append(self._node(child, n))
        for tu in node.transitions:
            t = tu.transition
            fn = AUDIO_TRANSITIONS.get(t.service)
            if fn is None:
                continue
            a, b = t.a_track, t.b_track
            if not (0 <= a < len(bufs) and 0 <= b < len(bufs)) or a == b:
                continue
            if bufs[b] is None:
                continue
            base = bufs[a] if bufs[a] is not None else np.zeros_like(bufs[b])
            prev = TransitionUse(t, tu.position - 1, tu.length) if tu.position > 0 else None
            bufs[a] = fn(self.ctx, base, bufs[b], tu, prev)
            bufs[b] = None
        for b in reversed(bufs):
            if b is not None:
                return b
        return None

    def _clip(self, node: ClipNode, n: int) -> np.ndarray | None:
        prod = node.producer
        if prod.service not in AUDIO_SERVICES:
            return None
        path = prod.resource_path
        if not path or not os.path.exists(path):
            return None
        info = probe(path)
        aidx = prod.props.get_int("audio_index", default=0) if prod.props.get("audio_index") else None
        if aidx == -1:
            return None
        a = info.audio_stream(aidx)
        if a is None:
            return None
        start = self.sample_start(node.frame)
        r = self.readers.get(node.key)
        if r is None:
            speed, pitch = 1.0, False
            if prod.service == "timewarp":
                speed = abs(prod.props.get_float("warp_speed", 1.0) or 1.0)
                pitch = prod.props.get_bool("warp_pitch")
            r = AudioReader(path, a.index, start, speed=speed, pitch_compensate=pitch)
            self.readers[node.key] = r
        self._last_used[node.key] = self._tick
        return r.read(start, n)
