"""MLT keyframe animation: a faithful Python port of mlt_animation.c.

A property such as ``"00:00:00.000=0.5;00:00:02.000~=1"`` is parsed into
keyframes.  The character before ``=`` selects the interpolation used from
that keyframe to the next one (``|`` discrete, ``~`` smooth, ``a``..``D``
easing curves, nothing = linear).  Positions before the first / after the last
keyframe hold the boundary value; negative positions count from the end.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

from .timecode import to_frames, to_float

# keyframe type ids
DISCRETE, LINEAR, SMOOTH_LOOSE, SMOOTH_NATURAL, SMOOTH_TIGHT = range(5)
_EASE_CHARS = "abcdefghijklmnopqrstuvwxyzABCD"   # sinusoidal_in .. bounce_in_out
_EASE_BASE = 100

_TYPE_CHARS = {"|": DISCRETE, "!": DISCRETE, "~": SMOOTH_LOOSE, "$": SMOOTH_NATURAL,
               "-": SMOOTH_TIGHT}
for _i, _c in enumerate(_EASE_CHARS):
    _TYPE_CHARS[_c] = _EASE_BASE + _i


def _type_from_char(c: str) -> int:
    if c and not c.isdigit():
        return _TYPE_CHARS.get(c, LINEAR)
    return LINEAR


# ---------------------------------------------------------------- interpolation

def _catmull_rom(x0, y0, x1, y1, x2, y2, x3, y3, t, alpha, tension):
    if x0 == x1:
        x0 -= 10000
    if x3 == x2:
        x3 += 10000
    dist = lambda xa, ya, xb, yb: math.hypot(xb - xa, yb - ya)
    m1 = m2 = 0.0
    t12 = dist(x1, y1, x2, y2) ** alpha
    if tension > 0.0 or (y0 > y1 > y2) or (y0 < y1 < y2):
        t01 = dist(x0, y0, x1, y1) ** alpha
        m1 = abs(tension) * (y2 - y1 + t12 * ((y1 - y0) / t01 - (y2 - y0) / (t01 + t12)))
    if tension > 0.0 or (y1 > y2 > y3) or (y1 < y2 < y3):
        t23 = dist(x2, y2, x3, y3) ** alpha
        m2 = abs(tension) * (y2 - y1 + t12 * ((y3 - y2) / t23 - (y3 - y1) / (t12 + t23)))
    a = 2.0 * (y1 - y2) + m1 + m2
    b = -3.0 * (y1 - y2) - m1 - m1 - m2
    return a * t ** 3 + b * t * t + m1 * t + y1


def _bounce_out(t):
    if t < 4 / 11.0:
        return (121 * t * t) / 16.0
    if t < 8 / 11.0:
        return (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17 / 5.0
    if t < 9 / 10.0:
        return (4356 / 361.0 * t * t) - (35442 / 1805.0 * t) + 16061 / 1805.0
    return (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268 / 25.0


def _ease_factor(kind: int, t: float) -> float:
    family, ease = divmod(kind, 3)          # ease: 0 in, 1 out, 2 in-out
    hp = math.pi / 2
    if family == 0:                          # sinusoidal
        return (math.sin((t - 1) * hp) + 1, math.sin(t * hp), 0.5 * (1 - math.cos(t * math.pi)))[ease]
    if family in (1, 2, 3, 4):               # quadratic..quintic
        n = family + 1
        if ease == 0:
            return t ** n
        if ease == 1:
            return 1 - (1 - t) ** n
        return (2 ** n) * t ** n / 2 if t < 0.5 else 1.0 - (-2 * t + 2) ** n / 2
    if family == 5:                          # exponential
        if t == 0.0:
            return 0.0
        if t == 1.0:
            return 1.0
        if ease == 0:
            return 2.0 ** (10 * t - 10)
        if ease == 1:
            return 1.0 - 2.0 ** (-10 * t)
        return 2 ** (20 * t - 10) / 2 if t < 0.5 else (2 - 2 ** (-20 * t + 10)) / 2
    if family == 6:                          # circular
        if ease == 0:
            return 1.0 - math.sqrt(max(0.0, 1.0 - t * t))
        if ease == 1:
            return math.sqrt(max(0.0, 1.0 - (t - 1.0) ** 2))
        if t < 0.5:
            return 0.5 * (1 - math.sqrt(max(0.0, 1 - 4 * t * t)))
        return 0.5 * (math.sqrt(max(0.0, -((2 * t) - 3) * ((2 * t) - 1))) + 1)
    if family == 7:                          # back
        f3 = lambda f: f * f * f - f * math.sin(f * math.pi)
        if ease == 0:
            return f3(t)
        if ease == 1:
            return 1 - f3(1 - t)
        return 0.5 * f3(2 * t) if t < 0.5 else 0.5 * (1 - f3(1 - (2 * t - 1))) + 0.5
    if family == 8:                          # elastic
        k = 13 * hp
        if ease == 0:
            return math.sin(k * t) * 2 ** (10 * (t - 1))
        if ease == 1:
            return math.sin(-k * (t + 1)) * 2 ** (-10 * t) + 1
        if t < 0.5:
            return 0.5 * math.sin(k * (2 * t)) * 2 ** (10 * ((2 * t) - 1))
        return 0.5 * (math.sin(-k * ((2 * t - 1) + 1)) * 2 ** (-10 * (2 * t - 1)) + 2)
    if family == 9:                          # bounce
        if ease == 0:
            return 1.0 - _bounce_out(1.0 - t)
        if ease == 1:
            return _bounce_out(t)
        return 0.5 * (1.0 - _bounce_out(1.0 - t * 2)) if t < 0.5 else 0.5 * _bounce_out(2.0 * t - 1.0) + 0.5
    return t


def interpolate_value(x0, y0, x1, y1, x2, y2, x3, y3, t, ktype):
    if ktype == DISCRETE:
        return y1
    if ktype == LINEAR:
        return y1 + (y2 - y1) * t
    if ktype == SMOOTH_LOOSE:
        return _catmull_rom(x0, y0, x1, y1, x2, y2, x3, y3, t, 0.0, 1.0)
    if ktype == SMOOTH_NATURAL:
        return _catmull_rom(x0, y0, x1, y1, x2, y2, x3, y3, t, 0.5, -1.0)
    if ktype == SMOOTH_TIGHT:
        return _catmull_rom(x0, y0, x1, y1, x2, y2, x3, y3, t, 0.5, 0.0)
    if ktype >= _EASE_BASE:
        return y1 + (y2 - y1) * _ease_factor(ktype - _EASE_BASE, t)
    return y1


# ---------------------------------------------------------------- value types

_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?%?")


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float
    o: float = 1.0
    percent: bool = False

    def as_tuple(self):
        return (self.x, self.y, self.w, self.h, self.o)


def parse_rect(s: str) -> Rect:
    nums = _NUM.findall(s.replace(",", "."))
    percent = "%" in s
    vals = []
    for n in nums:
        v = float(n.rstrip("%"))
        vals.append(v / 100.0 if n.endswith("%") else v)
    while len(vals) < 4:
        vals.append(0.0)
    o = vals[4] if len(vals) > 4 else 1.0
    return Rect(vals[0], vals[1], vals[2], vals[3], o, percent)


_COLOR_NAMES = {"black": 0x000000FF, "white": 0xFFFFFFFF, "red": 0xFF0000FF,
                "green": 0x00FF00FF, "blue": 0x0000FFFF, "transparent": 0x00000000}


def parse_color(s: str) -> tuple[int, int, int, int]:
    """Return (r, g, b, a) 0-255 from an MLT colour string."""
    s = (s or "").strip()
    v = None
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 8:           # #AARRGGBB
            a, r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4, 6))
            return r, g, b, a
        if len(h) >= 6:
            r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
            return r, g, b, 255
    elif s.lower().startswith("0x"):
        v = int(s[2:], 16)        # 0xRRGGBBAA
    elif s.lower() in _COLOR_NAMES:
        v = _COLOR_NAMES[s.lower()]
    else:
        try:
            v = int(s)
        except ValueError:
            v = 0x000000FF
    return (v >> 24) & 255, (v >> 16) & 255, (v >> 8) & 255, v & 255


def is_color(s: str) -> bool:
    s = s.strip()
    return s.startswith("#") or s.lower().startswith("0x")


# ---------------------------------------------------------------- animation

@dataclass
class _Key:
    frame: int
    ktype: int
    value: str


class Animation:
    """A parsed animated property. ``get_*`` methods evaluate at a position."""

    def __init__(self, data: str, fps: float, length: int):
        self.data = data
        self.fps = fps
        self.length = length
        self.keys: list[_Key] = []
        self.animated = False
        self._parse()

    def _parse(self):
        data = self.data or ""
        if "=" not in data:
            # Plain static value
            self.keys = [_Key(0, LINEAR, data)]
            return
        self.animated = True
        for token in data.split(";"):
            if not token.strip():
                continue
            if "=" in token:
                eq = token.index("=")
                time_part = token[:eq]
                ktype = _type_from_char(token[eq - 1]) if eq > 0 else LINEAR
                frame = to_frames(time_part.rstrip("|!~$-" + _EASE_CHARS) or "0", self.fps)
                value = token[eq + 1:]
                if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                    value = value[1:-1]
            else:
                frame, ktype, value = 0, LINEAR, token
            if frame < 0:
                frame += self.length
            self.keys.append(_Key(frame, ktype, value))
        # MLT keeps nodes sorted by frame; a later duplicate replaces the earlier one.
        uniq: dict[int, _Key] = {}
        for k in self.keys:
            uniq[k.frame] = k
        self.keys = [uniq[f] for f in sorted(uniq)]
        if not self.keys:
            self.keys = [_Key(0, LINEAR, "")]

    def _locate(self, position: int):
        keys = self.keys
        i = 0
        while i + 1 < len(keys) and position >= keys[i + 1].frame:
            i += 1
        return i

    def _eval(self, position: int, conv, interp_components):
        keys = self.keys
        i = self._locate(position)
        node = keys[i]
        if position <= node.frame or i + 1 >= len(keys) or node.ktype == DISCRETE:
            return conv(node.value)
        p = [keys[i - 1] if i > 0 else node, node, keys[i + 1],
             keys[i + 2] if i + 2 < len(keys) else keys[i + 1]]
        t = (position - p[1].frame) / (p[2].frame - p[1].frame)
        vals = [conv(k.value) for k in p]
        return interp_components([k.frame for k in p], vals, t, node.ktype)

    def get_double(self, position: int) -> float:
        def interp(xs, vs, t, kt):
            return interpolate_value(xs[0], vs[0], xs[1], vs[1], xs[2], vs[2], xs[3], vs[3], t, kt)
        return self._eval(position, lambda s: to_float(s), interp)

    def get_rect(self, position: int) -> Rect:
        def interp(xs, vs, t, kt):
            comps = []
            for c in range(5):
                ys = [v.as_tuple()[c] for v in vs]
                comps.append(interpolate_value(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2], xs[3], ys[3], t, kt))
            return Rect(*comps, percent=vs[1].percent)
        return self._eval(position, parse_rect, interp)

    def get_color(self, position: int) -> tuple[int, int, int, int]:
        def interp(xs, vs, t, kt):
            out = []
            for c in range(4):
                ys = [v[c] for v in vs]
                y = interpolate_value(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2], xs[3], ys[3], t, kt)
                out.append(int(min(255, max(0, y))))
            return tuple(out)
        return self._eval(position, parse_color, interp)

    def get_string(self, position: int) -> str:
        return self.keys[self._locate(position)].value


@lru_cache(maxsize=4096)
def animation(data: str, fps: float, length: int) -> Animation:
    return Animation(data, fps, length)
