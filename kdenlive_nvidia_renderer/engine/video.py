"""GPU video executor: turns timeline plans into output frames."""

from __future__ import annotations

import os

import cupy as cp
import numpy as np

from . import gpu
from .context import RenderContext
from .effects import (AUDIO_TRANSITIONS, IGNORED, VIDEO_FILTERS, VIDEO_TRANSITIONS, AUDIO_FILTERS,
                      Layer, canvas_layer, to_canvas, t_qtblend)
from .media import VideoReader, VideoRequest, load_image_rgba, probe
from .timeline import ClipNode, Node, StackNode

MEDIA_SERVICES = {"avformat", "avformat-novalidate", "timewarp", "qimage", "pixbuf"}
IMAGE_SERVICES = {"qimage", "pixbuf"}
GEOMETRY_FILTERS = {"qtblend", "affine", "pillar_echo", "crop"}


class VideoExecutor:
    def __init__(self, ctx: RenderContext):
        self.ctx = ctx
        self.readers: dict[tuple, VideoReader] = {}
        self._last_used: dict[tuple, int] = {}
        self._images: dict[str, Layer] = {}
        self._tick = 0

    # ------------------------------------------------------------------ public
    def render(self, node: Node | None) -> cp.ndarray:
        """Return the final output canvas (float RGBA, output size)."""
        self._tick += 1
        layer = self._node(node) if node is not None else None
        if layer is None:
            return gpu.new_canvas(self.ctx.out_w, self.ctx.out_h)
        return to_canvas(self.ctx, layer)

    def prefetch(self, node: Node | None):
        """Start decoders for clips that will be needed soon (hides FFmpeg/NVDEC start-up)."""
        if node is None:
            return
        if isinstance(node, StackNode):
            for i, c in enumerate(node.tracks):
                if c is not None and not node.hide_video[i]:
                    self.prefetch(c)
        elif isinstance(node, ClipNode) and not node.no_video:
            if node.producer.service in MEDIA_SERVICES - IMAGE_SERVICES and node.key not in self.readers:
                try:
                    self._reader(node)
                except Exception:
                    pass   # errors will surface when the frame is actually rendered

    def gc(self, keep_frames: int):
        """Close decoders that have not been used recently."""
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

    # ------------------------------------------------------------------ nodes
    def _node(self, node: Node) -> Layer | None:
        if node.no_video:
            return None
        if isinstance(node, ClipNode):
            layer = self._clip(node)
        elif isinstance(node, StackNode):
            layer = self._stack(node)
        else:
            layer = None
        if layer is None:
            return None
        for use in node.filters:
            svc = use.filter.service
            fn = VIDEO_FILTERS.get(svc)
            if fn is None:
                if svc not in IGNORED and svc not in AUDIO_FILTERS:
                    self.ctx.warn(f"Skipping unsupported video effect '{use.filter.kdenlive_id}' ({svc})")
                continue
            layer = fn(self.ctx, layer, use)
        return layer

    def _stack(self, node: StackNode) -> Layer | None:
        layers: list[Layer | None] = []
        for i, child in enumerate(node.tracks):
            if child is None or node.hide_video[i]:
                layers.append(None)
            else:
                layers.append(self._node(child))
        for tu in node.transitions:
            t = tu.transition
            a, b = t.a_track, t.b_track
            if not (0 <= a < len(layers) and 0 <= b < len(layers)) or a == b:
                continue
            svc = t.service
            if svc in AUDIO_TRANSITIONS and svc not in VIDEO_TRANSITIONS:
                continue
            if layers[b] is None:
                continue
            fn = VIDEO_TRANSITIONS.get(svc)
            if fn is None:
                self.ctx.warn(f"Unsupported transition '{t.kdenlive_id}' ({svc}); composited normally instead")
                fn = t_qtblend
            layers[a] = fn(self.ctx, layers[a], layers[b], tu)
            layers[b] = None
        for layer in reversed(layers):
            if layer is not None:
                return layer
        return None

    # ------------------------------------------------------------------ producers
    def _clip(self, node: ClipNode) -> Layer | None:
        prod = node.producer
        svc = prod.service
        ctx = self.ctx
        if svc in ("color", "colour"):
            from ..mlt.animation import animation
            res = prod.props.get("resource") or prod.props.get("colour") or "black"
            r, g, b, a = animation(res, ctx.fps, max(1, prod.length)).get_color(node.frame)
            return canvas_layer(ctx, gpu.solid(ctx.out_w, ctx.out_h, (r / 255, g / 255, b / 255, a / 255)))
        if svc in MEDIA_SERVICES:
            path = prod.resource_path
            if not path or not os.path.exists(path):
                ctx.warn(f"Missing media file: {path or prod.resource}")
                return None
            info = probe(path)
            vidx = prod.props.get_int("video_index", default=0) if prod.props.get("video_index") else None
            if vidx == -1:
                return None
            v = info.video_stream(vidx)
            if v is None:
                return None
            if svc in IMAGE_SERVICES or v.is_image:
                return self._image(path, v)
            reader = self._reader(node)
            raw = reader.read(node.frame)
            if raw is None:
                return None
            self._last_used[node.key] = self._tick
            req = reader.req
            gpu_raw = cp.asarray(raw)
            img = gpu.yuv420_to_rgba(gpu_raw, req.out_w, req.out_h, reader.is16, v.colorspace, v.full_range)
            layer = self._orient(img, v)
            layer.fmt = "yuv"
            return layer
        if svc == "kdenlivetitle":
            ctx.warn("Title clips are not supported yet and are left out of the render")
            return None
        ctx.warn(f"Unsupported clip type '{svc}' ({prod.resource}) is left out of the render")
        return None

    def _orient(self, img: cp.ndarray, v) -> Layer:
        sar = float(v.sar) if v.sar else 1.0
        h, w = img.shape[:2]
        dar = (w * sar) / h
        rot = v.rotation % 360
        if rot in (90, 270):
            out = cp.empty((w, h, 4), dtype=cp.float32)
            if rot == 90:      # clockwise
                T = np.array([[0, -1, h], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
            else:
                T = np.array([[0, 1, 0], [-1, 0, w], [0, 0, 1]], dtype=np.float64)
            gpu.affine_composite(out, img, T, mode=-1, nearest=True)
            return Layer(out, 1.0 / dar)
        if rot == 180:
            return Layer(cp.ascontiguousarray(img[::-1, ::-1]), dar)
        return Layer(img, dar)

    def _image(self, path: str, v) -> Layer:
        cached = self._images.get(path)
        if cached is None:
            raw, w, h = load_image_rgba(path)
            img = gpu.rgba8_to_rgba(cp.asarray(raw), w, h)
            cached = self._orient(img, v)
            self._images[path] = cached
        # effects modify layers in place, so hand out a copy
        return Layer(cached.img.copy(), cached.dar)

    def _decode_size(self, node: ClipNode, v) -> tuple[int, int]:
        """Decode at the size the picture is displayed at (never larger than native)."""
        w, h = v.width, v.height
        needs_native = any(u.filter.service in GEOMETRY_FILTERS for u in node.filters)
        if needs_native:
            return w, h
        sar = float(v.sar) if v.sar else 1.0
        dw, dh = (h, w * sar) if v.rotation % 180 == 90 else (w * sar, h)   # display size in px
        out_w, out_h = self.ctx.out_w, self.ctx.out_h
        k = min(out_w / dw, out_h / dh)
        if k >= 1.0:
            return w, h
        nw = max(2, int(round(w * k / 2)) * 2)
        nh = max(2, int(round(h * k / 2)) * 2)
        return min(nw, w), min(nh, h)

    def _reader(self, node: ClipNode) -> VideoReader:
        r = self.readers.get(node.key)
        if r is not None:
            return r
        prod = node.producer
        info = probe(prod.resource_path)
        vidx = prod.props.get_int("video_index", default=0) if prod.props.get("video_index") else None
        v = info.video_stream(vidx)
        speed = 1.0
        if prod.service == "timewarp":
            speed = prod.props.get_float("warp_speed", 1.0) or 1.0
            if speed < 0:
                self.ctx.warn("Reverse playback (negative speed) is not supported; playing forward")
                speed = abs(speed)
        ow, oh = self._decode_size(node, v)
        fps = self.ctx.fps_fraction
        same_rate = v.fps and abs(float(v.fps) - float(fps)) / float(fps) < 1e-4
        req = VideoRequest(path=prod.resource_path, stream=v.index, out_w=ow, out_h=oh, fps=fps,
                           speed=speed, same_rate=bool(same_rate))
        r = VideoReader(req, v, start=node.frame, log=self.ctx.log)
        self.readers[node.key] = r
        self._last_used[node.key] = self._tick
        return r
