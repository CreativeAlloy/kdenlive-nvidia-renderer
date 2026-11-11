"""Effect and transition implementations.

Every handler is a port of the corresponding MLT / frei0r / FFmpeg code path
that Kdenlive uses, executed with CUDA kernels from :mod:`gpu`.

Video filters:  ``fn(ctx, layer, use) -> Layer``
Audio filters:  ``fn(ctx, samples, use, frame) -> samples`` (numpy, float32 (n, 2))
Video transitions: ``fn(ctx, a_layer, b_layer, use) -> Layer`` (result replaces track A)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..mlt.animation import Rect, animation, parse_color
from .timeline import FilterUse, TransitionUse

try:
    import cupy as cp
    from . import gpu
except Exception:          # analysis-only mode (no GPU stack available)
    cp = None
    gpu = None


# ============================================================================ layer

@dataclass
class Layer:
    """A picture travelling through the pipeline.

    ``img`` is float32 RGBA (H, W, 4).  ``dar`` is the display aspect ratio of
    the picture content.  ``canvas`` is True when the picture already is a
    full output-frame canvas (identity placement).
    """
    img: "cp.ndarray"
    dar: float
    canvas: bool = False
    fmt: str = "rgba"      # what MLT would hold this picture as: "yuv" or "rgba"

    @property
    def w(self):
        return self.img.shape[1]

    @property
    def h(self):
        return self.img.shape[0]


def fit_transform(ctx, layer: Layer, rect: tuple[float, float, float, float], distort=False,
                  rotation=0.0, anchor: tuple[float, float] | None = None) -> np.ndarray:
    """3x3 matrix mapping layer pixels into output pixels for an MLT/Qt "rect".

    ``rect`` is in *profile* pixels; the result accounts for output scaling.
    Mirrors build_transform() of MLT's qtblend (fit + centre, optional rotation).
    """
    x, y, w, h = rect
    T = np.eye(3)

    def tr(dx, dy):
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)

    def sc(sx, sy):
        return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)

    def rot(deg):
        r = math.radians(deg)
        c, s = math.cos(r), math.sin(r)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    # profile px -> output px
    T = sc(ctx.sx, ctx.sy) @ tr(x, y)
    if rotation:
        if anchor is not None:
            ax, ay = anchor[0] * w, anchor[1] * h
            T = T @ tr(ax, ay) @ rot(rotation) @ tr(-ax, -ay)
        else:
            T = T @ rot(rotation)
    # Work in "profile pixel" units for the box: content has ``layer.dar`` display aspect,
    # profile pixels have sample aspect ``ctx.profile_sar``.
    if distort:
        box_w, box_h = w, h
    else:
        resize_dar = w * ctx.profile_sar / h if h else 1.0
        if layer.dar >= resize_dar:
            box_w = w
            box_h = w * ctx.profile_sar / layer.dar
        else:
            box_h = h
            box_w = h * layer.dar / ctx.profile_sar
        T = T @ tr((w - box_w) / 2.0, (h - box_h) / 2.0)
    T = T @ sc(box_w / layer.w, box_h / layer.h)
    return T


def to_canvas(ctx, layer: Layer | None) -> "cp.ndarray":
    """Return a full output-frame canvas for a layer (fit + letterbox, transparent padding)."""
    if layer is None:
        return gpu.new_canvas(ctx.out_w, ctx.out_h)
    if layer.canvas and layer.w == ctx.out_w and layer.h == ctx.out_h:
        return layer.img
    canvas = cp.empty((ctx.out_h, ctx.out_w, 4), dtype=cp.float32)
    T = fit_transform(ctx, layer, (0, 0, ctx.profile_w, ctx.profile_h))
    gpu.affine_composite(canvas, layer.img, T, mode=-1)
    return canvas


def canvas_layer(ctx, img, fmt: str = "rgba") -> Layer:
    return Layer(img, ctx.frame_dar, canvas=True, fmt=fmt)


def layer_scale(ctx, layer: Layer) -> float:
    """Layer pixels per profile pixel when the layer is shown full-frame."""
    if layer.canvas:
        return layer.w / ctx.profile_w
    if layer.dar >= ctx.frame_dar:
        box_w = ctx.profile_w
    else:
        box_w = ctx.profile_h * layer.dar / ctx.profile_sar
    return layer.w / box_w


# ============================================================================ param helpers

def _prop(use, *names):
    props = use.filter.props if isinstance(use, FilterUse) else use.transition.props
    for n in names:
        v = props.get(n)
        if v is not None and v != "":
            return v
    return None


def p_double(ctx, use, names, default: float) -> float:
    if isinstance(names, str):
        names = (names,)
    v = _prop(use, *names)
    if v is None:
        return default
    return animation(v, ctx.fps, use.length).get_double(use.position)


def p_rect(ctx, use, name: str) -> Rect | None:
    v = _prop(use, name)
    if v is None:
        return None
    r = animation(v, ctx.fps, use.length).get_rect(use.position)
    return r


def p_int(use, name, default=0) -> int:
    v = _prop(use, name)
    try:
        return int(float(v)) if v is not None else default
    except ValueError:
        return default


def p_bool(use, name, default=False) -> bool:
    v = _prop(use, name)
    if v is None:
        return default
    try:
        return float(v) != 0
    except ValueError:
        return default


def _clamp_u8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


# ============================================================================ registries

VIDEO_FILTERS: dict[str, callable] = {}
AUDIO_FILTERS: dict[str, callable] = {}
VIDEO_TRANSITIONS: dict[str, callable] = {}
AUDIO_TRANSITIONS: dict[str, callable] = {}
# Filters that have no visible/audible effect when rendering (meters, internal helpers)
IGNORED = {"audiolevel", "loudness_meter", "audiospectrum", "audiowaveform", "dance", "fft"}


def vfilter(*names, rgba=True):
    """Register a video filter. ``rgba``: MLT runs it on an RGBA image (format switches)."""
    def deco(fn):
        def wrapped(ctx, layer, use):
            out = fn(ctx, layer, use)
            if rgba and out is not None:
                out.fmt = "rgba"
            return out
        for n in names:
            VIDEO_FILTERS[n] = wrapped
        return fn
    return deco


def afilter(*names):
    def deco(fn):
        for n in names:
            AUDIO_FILTERS[n] = fn
        return fn
    return deco


def vtransition(*names):
    def deco(fn):
        for n in names:
            VIDEO_TRANSITIONS[n] = fn
        return fn
    return deco


def atransition(*names):
    def deco(fn):
        for n in names:
            AUDIO_TRANSITIONS[n] = fn
        return fn
    return deco


def is_audio_service(service: str) -> bool:
    return service in AUDIO_FILTERS


# ============================================================================ video filters

@vfilter("frei0r.brightness")
def f_frei0r_brightness(ctx, layer, use):
    b = p_double(ctx, use, ("Brightness", "0"), 0.5)
    val = int((b - 0.5) * 512.0)
    if val == 0:
        return layer
    i = np.arange(256, dtype=np.int64)
    lut = (i * (256 + val)) >> 8 if val < 0 else i + (((256 - i) * val) >> 8)
    gpu.apply_lut3(layer.img, np.tile(_clamp_u8(lut), (3, 1)))
    return layer


@vfilter("frei0r.contrast0r")
def f_frei0r_contrast(ctx, layer, use):
    c = p_double(ctx, use, ("Contrast", "0"), 0.5)
    val = int((c - 0.5) * 512.0)
    if val == 0:
        return layer
    i = np.arange(256, dtype=np.int64)
    lut = np.where(i < 128, i - (((128 - i) * val) >> 8), i + (((i - 128) * val) >> 8))
    gpu.apply_lut3(layer.img, np.tile(_clamp_u8(lut), (3, 1)))
    return layer


@vfilter("frei0r.saturat0r")
def f_frei0r_saturat0r(ctx, layer, use):
    s = p_double(ctx, use, ("Saturation", "0"), 1.0 / 8.0) * 8.0
    gpu.saturat0r(layer.img, s)
    return layer


@vfilter("avfilter.exposure")
def f_av_exposure(ctx, layer, use):
    exposure = np.float32(p_double(ctx, use, "av.exposure", 0.0))
    black = np.float32(p_double(ctx, use, "av.black", 0.0))
    diff = np.float32(abs(np.float32(2.0) ** (-exposure) - black))
    diff = diff if diff > 0 else np.float32(1.0 / 1024.0)
    scale = np.float32(1.0) / diff
    i = np.arange(256, dtype=np.float32) / np.float32(255.0)
    v = (i - black) * scale
    lut = _clamp_u8(np.rint(np.clip(v, 0, 1) * 255.0))
    gpu.apply_lut3(layer.img, np.tile(lut, (3, 1)))
    return layer


@vfilter("avfilter.colorlevels")
def f_av_colorlevels(ctx, layer, use):
    luts = []
    for c in "rgba":
        imin = int(round(p_double(ctx, use, f"av.{c}imin", 0.0) * 255))
        imax = int(round(p_double(ctx, use, f"av.{c}imax", 1.0) * 255))
        omin = int(round(p_double(ctx, use, f"av.{c}omin", 0.0) * 255))
        omax = int(round(p_double(ctx, use, f"av.{c}omax", 1.0) * 255))
        imin = max(imin, 0)          # auto (-1) levels are not supported; treat as 0/255
        imax = imax if imax >= 0 else 255
        coeff = np.float32((omax - omin) / float(imax - imin)) if imax != imin else np.float32(0)
        i = np.arange(256, dtype=np.float32)
        val = (i - np.float32(imin)) * coeff + np.float32(omin)
        luts.append(_clamp_u8(np.trunc(val).astype(np.int64)))
    gpu.apply_lut3(layer.img, np.stack(luts[:3]))
    if not np.array_equal(luts[3], np.arange(256)):
        gpu.apply_alpha_lut(layer.img, luts[3])
    return layer


@vfilter("brightness", rgba=False)
def f_brightness(ctx, layer, use):
    if _prop(use, "level") is not None:
        level = p_double(ctx, use, "level", 1.0)
    else:
        level = abs(p_double(ctx, use, "start", 1.0))
        if _prop(use, "end") is not None:
            end = abs(p_double(ctx, use, "end", 1.0))
            progress = use.position / (use.length - 1) if use.length > 1 else 1.0
            level += (end - level) * progress
    alpha_level = p_double(ctx, use, "alpha", 1.0) if _prop(use, "alpha") is not None else 1.0
    if alpha_level < 0.0:
        alpha_level = level
    i = np.arange(256, dtype=np.float64)
    if level != 1.0:
        if layer.fmt == "yuv":
            gpu.brightness_yuv(layer.img, level, ctx.colorspace)
        else:
            gpu.apply_lut3(layer.img, np.tile(_clamp_u8(np.round(i * level)), (3, 1)))
    if alpha_level != 1.0:
        gpu.apply_alpha_lut(layer.img, _clamp_u8(np.round(i * alpha_level)))
    return layer


@vfilter("qtblend")
def f_qtblend(ctx, layer, use):
    """Kdenlive 'Transform' effect: place the picture into rect on a transparent frame."""
    r = p_rect(ctx, use, "rect")
    if r is None:
        rect = (0.0, 0.0, float(ctx.profile_w), float(ctx.profile_h), 1.0)
    else:
        rect = r.as_tuple()
        if r.percent:
            rect = (rect[0] * ctx.profile_w, rect[1] * ctx.profile_h, rect[2] * ctx.profile_w,
                    rect[3] * ctx.profile_h, rect[4])
    rotation = p_double(ctx, use, "rotation", 0.0)
    anchor = None
    if rotation:
        ra = p_rect(ctx, use, "rotate_anchor")
        if ra is not None:
            anchor = (ra.x, ra.y)
        elif p_bool(use, "rotate_center", False):
            anchor = (0.5, 0.5)
    distort = p_bool(use, "distort")
    mode = p_int(use, "compositing", 0)
    x, y, w, h, o = rect
    if (not rotation and o >= 1.0 and mode == 0 and layer.canvas
            and (x, y, w, h) == (0, 0, ctx.profile_w, ctx.profile_h)):
        return layer
    bg = _prop(use, "background_color")
    if bg:
        rr, gg, bb, aa = parse_color(bg)
        out = gpu.solid(ctx.out_w, ctx.out_h, (rr / 255, gg / 255, bb / 255, aa / 255))
    else:
        out = gpu.new_canvas(ctx.out_w, ctx.out_h)
    if w > 0 and h > 0:
        T = fit_transform(ctx, layer, (x, y, w, h), distort=distort, rotation=rotation, anchor=anchor)
        gpu.affine_composite(out, layer.img, T, opacity=max(0.0, min(1.0, o)), mode=mode)
    return canvas_layer(ctx, out)


@vfilter("avfilter.hflip", "frei0r.flippo_h", rgba=False)
def f_hflip(ctx, layer, use):
    layer.img = cp.ascontiguousarray(layer.img[:, ::-1])
    return layer


@vfilter("avfilter.vflip", rgba=False)
def f_vflip(ctx, layer, use):
    layer.img = cp.ascontiguousarray(layer.img[::-1])
    return layer



def _scale0tilt_geometry(w: int, h: int, cl, cr, ct, cb, sx, sy, tx, ty):
    """update_scaler() of frei0r scale0tilt: returns (src_rect, dst_rect) or None when nothing is drawn.

    Parameters are the plugin's internal values (scale already doubled, tilt already mapped to -1..1).
    The C code keeps the rectangles in ``float``, so this does too.
    """
    f32 = np.float32
    src_x, src_y = f32(w * cl), f32(h * ct)
    src_w, src_h = f32(w * (1.0 - cl - cr)), f32(h * (1.0 - ct - cb))
    dst_x, dst_y = f32(w * cl * sx + tx * w), f32(h * ct * sy + ty * h)
    dst_w, dst_h = f32(w * (1.0 - cl - cr) * sx), f32(h * (1.0 - ct - cb) * sy)
    eps = 1e-6
    if dst_w < eps or dst_h < eps or src_w < eps or src_h < eps:
        return None
    fw, fh = f32(w), f32(h)
    if dst_x + dst_w > fw:
        src_w = f32(src_w * ((fw - dst_x) / dst_w))
        dst_w = f32(fw - dst_x)
    if dst_y + dst_h > fh:
        src_h = f32(src_h * ((fh - dst_y) / dst_h))
        dst_h = f32(fh - dst_y)
    if dst_x < 0:
        src_x = f32(src_x - dst_x * (src_w / dst_w))
        src_w = f32(src_w * ((dst_w + dst_x) / dst_w))
        dst_w = f32(dst_w + dst_x)
        dst_x = f32(0)
    if dst_y < 0:
        src_y = f32(src_y - dst_y * (src_h / dst_h))
        src_h = f32(src_h * ((dst_h + dst_y) / dst_h))
        dst_h = f32(dst_h + dst_y)
        dst_y = f32(0)
    if dst_w < eps or dst_h < eps or src_w < eps or src_h < eps:
        return None

    def lroundf(v):      # C lroundf: half away from zero
        v = float(v)
        return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))

    return ((float(src_x), float(src_y), float(src_w), float(src_h)),
            (lroundf(dst_x), lroundf(dst_y), lroundf(dst_w), lroundf(dst_h)))


@vfilter("frei0r.scale0tilt")
def f_frei0r_scale0tilt(ctx, layer, use):
    """Crop, Scale and Tilt (frei0r scale0tilt): crop, then scale and place the picture; the rest is transparent."""
    def p(name, index, default):
        return p_double(ctx, use, (name, str(index)), default)
    cl, cr = p("Clip left", 0, 0.0), p("Clip right", 1, 0.0)
    ct, cb = p("Clip top", 2, 0.0), p("Clip bottom", 3, 0.0)
    sx, sy = p("Scale X", 4, 0.5) * 2.0, p("Scale Y", 5, 0.5) * 2.0
    tx, ty = p("Tilt X", 6, 0.5) * 2.0 - 1.0, p("Tilt Y", 7, 0.5) * 2.0 - 1.0
    geo = _scale0tilt_geometry(layer.w, layer.h, cl, cr, ct, cb, sx, sy, tx, ty)
    if geo is None:
        layer.img = cp.zeros_like(layer.img)
    else:
        layer.img = gpu.resample_rect(layer.img, *geo)
    return layer


@vfilter("avfilter.gblur")
def f_gblur(ctx, layer, use):
    sigma = p_double(ctx, use, "av.sigma", 0.5)
    sigma_v = p_double(ctx, use, "av.sigmaV", -1.0)
    if sigma_v >= 0 and abs(sigma_v - sigma) > 1e-6:
        ctx.warn("Gaussian blur with a different vertical sigma is approximated (uses horizontal sigma)")
    # MLT blurs the profile-sized frame; convert sigma to this layer's pixel scale.
    layer.img = gpu.gaussian_blur(layer.img, sigma * layer_scale(ctx, layer))
    return layer


# ============================================================================ audio filters

def _frame_ramp(prev: float, cur: float, n: int) -> np.ndarray:
    if prev == cur:
        return np.full((n, 1), cur, dtype=np.float32)
    return (prev + (cur - prev) * np.arange(n, dtype=np.float32) / n).reshape(-1, 1)


def _prev_use(use):
    if use.position <= 0:
        return use
    return type(use)(use.filter, use.position - 1, use.length)


def _volume_gain(ctx, use) -> float:
    if _prop(use, "level") is not None:
        return 10 ** (p_double(ctx, use, "level", 0.0) / 20.0)
    gain_s = _prop(use, "gain")
    gain = 1.0
    if gain_s is not None and not gain_s.lower().startswith("normali"):
        g = gain_s.strip().lower()
        if g.endswith("db"):
            gain = 10 ** (float(g[:-2]) / 20.0)
        else:
            gain = abs(float(g) if g else 1.0)
        end_s = _prop(use, "end")
        if end_s is not None:
            e = end_s.strip().lower()
            end = 10 ** (float(e[:-2]) / 20.0) if e.endswith("db") else abs(float(e or 0))
            progress = use.position / (use.length - 1) if use.length > 1 else 1.0
            gain += (end - gain) * progress
    elif gain_s is not None:
        ctx.warn("Audio normalisation (volume 'normalise') is not supported; applied as unity gain")
    mx = _prop(use, "max_gain")
    if mx:
        m = mx.strip().lower()
        mval = 10 ** (float(m[:-2]) / 20.0) if m.endswith("db") else abs(float(m))
        if mval > 0 and gain > mval:
            gain = mval
    return gain


@afilter("volume")
def a_volume(ctx, samples, use, frame):
    cur = _volume_gain(ctx, use)
    prev = _volume_gain(ctx, _prev_use(use))
    if cur == 1.0 and prev == 1.0:
        return samples
    return samples * _frame_ramp(prev, cur, len(samples))


@afilter("panner")
def a_panner(ctx, samples, use, frame):
    # MLT panner: start in 0..1 (0.5 centre), channel -1 = stereo balance
    pos = p_double(ctx, use, ("start", "split"), 0.5)
    channel = p_int(use, "channel", -1)
    if abs(pos - 0.5) < 1e-9:
        return samples
    if channel == -1:      # balance
        left = min(1.0, 2.0 * (1.0 - pos))
        right = min(1.0, 2.0 * pos)
        return samples * np.array([[left, right]], dtype=np.float32)
    if channel in (0, 1):  # pan one channel across both
        src = samples[:, channel:channel + 1]
        out = samples.copy()
        out[:, 0:1] = src * (1.0 - pos)
        out[:, 1:2] = src * pos
        return out
    return samples



def _ffmpeg_double_option(use, name: str, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """An FFmpeg double option as MLT's avfilter wrapper sets it: a value that does not parse or lies
    outside the option's range is rejected by av_opt_set(), so the filter keeps its own default."""
    v = _prop(use, name)
    if v is None:
        return default
    try:
        x = float(v)
    except ValueError:
        return default
    return x if lo <= x <= hi else default


@afilter("avfilter.deesser")
def a_deesser(ctx, samples, use, frame):
    """FFmpeg 'deesser' (bit-exact port).  The filter is a running recurrence: like MLT's avfilter
    wrapper, one instance keeps its state for the whole render."""
    from . import audio_gpu
    intensity = _ffmpeg_double_option(use, "av.i", 0.0)
    max_amount = _ffmpeg_double_option(use, "av.m", 0.5)
    frequency = _ffmpeg_double_option(use, "av.f", 0.5)
    mode_s = (_prop(use, "av.s") or "o").strip().lower()
    mode = {"i": 0, "in": 0, "0": 0, "o": 1, "out": 1, "1": 1, "e": 2, "ess": 2, "2": 2}.get(mode_s, 1)
    states = ctx.__dict__.setdefault("filter_states", {})
    key = id(use.filter)
    st = states.get(key)
    if st is None or st.shape[0] != samples.shape[1]:
        st = states[key] = audio_gpu.new_deesser_state(samples.shape[1])
    return audio_gpu.deesser(samples, st, intensity, max_amount, frequency, mode, 48000)


@afilter("audiomap", "channelcopy", "mono")
def a_channels(ctx, samples, use, frame):
    svc = use.filter.service
    if svc == "mono":
        m = samples.mean(axis=1, keepdims=True)
        return np.repeat(m, 2, axis=1)
    if svc == "channelcopy":
        frm = p_int(use, "from", 0)
        to = p_int(use, "to", 1)
        out = samples.copy()
        if 0 <= frm < 2 and 0 <= to < 2:
            if p_bool(use, "swap"):
                out[:, frm], out[:, to] = samples[:, to], samples[:, frm]
            else:
                out[:, to] = samples[:, frm]
        return out
    return samples


AUDIO_FILTERS.pop("audiomap", None)


# ============================================================================ transitions

def _transition_progress(use: TransitionUse) -> float:
    return use.position / use.length if use.length > 0 else 0.0


@vtransition("qtblend")
def t_qtblend(ctx, a: Layer | None, b: Layer, use):
    canvas = to_canvas(ctx, a)
    r = p_rect(ctx, use, "rect")
    if r is None:
        rect = (0.0, 0.0, float(ctx.profile_w), float(ctx.profile_h), 1.0)
    else:
        rect = r.as_tuple()
        if r.percent:
            rect = (rect[0] * ctx.profile_w, rect[1] * ctx.profile_h, rect[2] * ctx.profile_w,
                    rect[3] * ctx.profile_h, rect[4])
    x, y, w, h, o = rect
    rotation = p_double(ctx, use, "rotation", 0.0)
    anchor = (0.5, 0.5) if (rotation and p_bool(use, "rotate_center")) else None
    mode = p_int(use, "compositing", 0)
    passthrough = (b.fmt == "yuv" and mode == 0 and o >= 1.0 and not rotation
                   and (x, y, w, h) == (0, 0, ctx.profile_w, ctx.profile_h)
                   and abs(b.dar - ctx.frame_dar) < 1e-3)
    if w > 0 and h > 0 and o > 0:
        T = fit_transform(ctx, b, (x, y, w, h), distort=p_bool(use, "distort"),
                          rotation=rotation, anchor=anchor)
        gpu.affine_composite(canvas, b.img, T, opacity=min(1.0, o), mode=mode)
    return canvas_layer(ctx, canvas, "yuv" if passthrough else "rgba")


@vtransition("composite", "affine", "frei0r.cairoblend", "frei0r.cairoaffineblend", "movit.overlay",
             "frei0r.addition", "frei0r.multiply", "frei0r.screen", "frei0r.overlay")
def t_composite(ctx, a, b, use):
    """MLT composite / affine family: geometry rect + opacity, optional luma wipe."""
    svc = use.transition.service
    canvas = to_canvas(ctx, a)
    geom_name = "geometry" if _prop(use, "geometry") is not None else (
        "rect" if _prop(use, "rect") is not None else "transition.rect")
    r = p_rect(ctx, use, geom_name)
    if r is None:
        x, y, w, h, o = 0.0, 0.0, float(ctx.profile_w), float(ctx.profile_h), 1.0
    else:
        x, y, w, h, o = r.as_tuple()
        if r.percent:
            x, y, w, h = x * ctx.profile_w, y * ctx.profile_h, w * ctx.profile_w, h * ctx.profile_h
        elif svc == "composite":
            o = o / 100.0          # composite geometry opacity is a percentage
    mode = {"frei0r.addition": 12, "frei0r.multiply": 13, "frei0r.screen": 14,
            "frei0r.overlay": 15}.get(svc, 0)
    if svc == "frei0r.cairoblend":
        blend = (_prop(use, "1", "blend mode") or "normal").lower()
        mode = {"add": 12, "multiply": 13, "screen": 14, "overlay": 15, "darken": 16, "lighten": 17,
                "colordodge": 18, "colorburn": 19, "hardlight": 20, "softlight": 21,
                "difference": 22, "exclusion": 23}.get(blend, 0)
        o = p_double(ctx, use, ("0", "opacity"), 1.0)
    luma = _prop(use, "luma")
    distort = p_bool(use, "distort")
    if luma and svc == "composite":
        b_canvas = gpu.new_canvas(ctx.out_w, ctx.out_h)
        T = fit_transform(ctx, b, (x, y, w, h), distort=distort)
        gpu.affine_composite(b_canvas, b.img, T, mode=-1)
        lmap = ctx.luma_map(luma, p_bool(use, "luma_invert"))
        if lmap is not None:
            gpu.luma_wipe(canvas, b_canvas, lmap, _transition_progress(use), p_double(ctx, use, "softness", 0.0))
            return canvas_layer(ctx, canvas)
    if w > 0 and h > 0 and o > 0:
        T = fit_transform(ctx, b, (x, y, w, h), distort=distort)
        gpu.affine_composite(canvas, b.img, T, opacity=min(1.0, o), mode=mode)
    return canvas_layer(ctx, canvas)


@vtransition("luma", "dissolve")
def t_luma(ctx, a, b, use):
    canvas = to_canvas(ctx, a)
    if a is not None and a.canvas and canvas is a.img:
        canvas = canvas.copy()
    b_canvas = to_canvas(ctx, b)
    mix = _transition_progress(use)
    if _prop(use, "fixed") is not None:
        mix = p_double(ctx, use, "fixed", mix)
    reverse = p_bool(use, "reverse")
    invert = p_bool(use, "invert")
    resource = _prop(use, "resource")
    softness = p_double(ctx, use, "softness", 0.0)
    if resource:
        lmap = ctx.luma_map(resource, False)
        if lmap is not None:
            rev = (not reverse) if invert else reverse
            m = 1 - mix if rev else mix
            if invert:
                dst, src = b_canvas.copy(), canvas
            else:
                dst, src = canvas, b_canvas
            if rev:
                lmap = 1.0 - lmap
            gpu.luma_wipe(dst, src, lmap, m, softness)
            return canvas_layer(ctx, dst)
    m = 1 - mix if reverse else mix
    gpu.dissolve(canvas, b_canvas, m)
    both_yuv = a is not None and a.fmt == "yuv" and b.fmt == "yuv"
    return canvas_layer(ctx, canvas, "yuv" if both_yuv else "rgba")


@atransition("mix")
def at_mix(ctx, a: np.ndarray, b: np.ndarray, use, prev_use):
    props = use.transition.props

    def level(u):
        if props.get("start") is None:
            return None
        start = props.get_float("start")
        mix = _transition_progress(u)
        if props.get("end") is not None:
            end = props.get_float("end")
            return start + (end - start) * mix
        if start >= 0:
            return start
        return mix

    cur = level(use)
    prev = level(prev_use) if prev_use is not None else cur
    reverse = props.get_bool("reverse")
    if props.get_bool("sum"):
        if cur is None:
            return a + b
        if reverse:
            cur, prev = 1 - cur, 1 - prev
        return a + b * _frame_ramp(prev, cur, len(a))
    if props.get_bool("combine"):
        return a + b
    if cur is None:
        cur = prev = 0.5
    if reverse:
        cur, prev = 1 - cur, 1 - prev
    ramp = _frame_ramp(prev, cur, len(a))
    if props.get_float("start") < -1.0:
        return a * np.cos(ramp * math.pi / 2) + b * np.sin(ramp * math.pi / 2)
    return a * (1 - ramp) + b * ramp


# ============================================================================ support checks

def video_filter_supported(service: str) -> bool:
    return service in VIDEO_FILTERS


def classify_filter(service: str) -> str:
    """'video', 'audio', 'ignored' or 'unsupported'."""
    if service in IGNORED:
        return "ignored"
    if service in VIDEO_FILTERS:
        return "video"
    if service in AUDIO_FILTERS:
        return "audio"
    return "unsupported"
