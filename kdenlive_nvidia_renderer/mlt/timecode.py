"""MLT time value parsing, mirroring mlt_property.c exactly.

MLT accepts positions as plain frame numbers, clock strings ("HH:MM:SS.mmm")
or SMPTE timecodes ("HH:MM:SS:FF" / "HH:MM:SS;FF").  The rounding rules are
deliberately odd (hours and minutes are floored separately) and must be
replicated bit-for-bit or clip boundaries drift by a frame.
"""

from __future__ import annotations

import math
import re

_NUM_PREFIX = re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def _strtod(s: str) -> float:
    """C strtod: parse the longest numeric prefix, 0.0 if none."""
    m = _NUM_PREFIX.match(s.replace(",", "."))
    return float(m.group(0)) if m else 0.0


def _atoi(s: str) -> int:
    m = re.match(r"^\s*[-+]?\d+", s)
    return int(m.group(0)) if m else 0


def _lrint(x: float) -> int:
    # C lrint with default rounding mode = round half to even (Python's round).
    return int(round(x))


def _time_clock_to_frames(s: str, fps: float) -> int:
    hours = minutes = 0
    pos = s.rfind(":")
    if pos >= 0:
        seconds = _strtod(s[pos + 1:])
        head = s[:pos]
        pos = head.rfind(":")
        if pos >= 0:
            minutes = _atoi(head[pos + 1:])
            hours = _atoi(head[:pos])
        else:
            minutes = _atoi(head)
    else:
        seconds = _strtod(s)
    return math.floor(fps * hours * 3600) + math.floor(fps * minutes * 60) + _lrint(fps * seconds)


def _time_code_to_frames(s: str, fps: float) -> int:
    hours = minutes = seconds = 0
    pos = s.rfind(";")
    if pos < 0:
        pos = s.rfind(":")
    if pos >= 0:
        frames = _atoi(s[pos + 1:])
        head = s[:pos]
        pos = head.rfind(":")
        if pos >= 0:
            seconds = _atoi(head[pos + 1:])
            head = head[:pos]
            pos = head.rfind(":")
            if pos >= 0:
                minutes = _atoi(head[pos + 1:])
                hours = _atoi(head[:pos])
            else:
                minutes = _atoi(head)
        else:
            seconds = _atoi(head)
    else:
        frames = _atoi(s)
    return (math.floor(fps * hours * 3600) + math.floor(fps * minutes * 60)
            + math.ceil(fps * seconds) + frames)


def to_frames(value: str | int | float | None, fps: float, default: int = 0) -> int:
    """Convert any MLT position value to an integer frame number."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    s = value.strip()
    if not s:
        return default
    if s[0] == "#":
        rgb = int(s[1:], 16)
        alpha = (rgb >> 24) if len(s) > 7 else 0xFF
        return ((rgb << 8) | alpha) & 0xFFFFFFFF
    if s.startswith("0x"):
        return int(s[2:], 16)
    if fps > 0 and ":" in s:
        if "." in s or "," in s:
            return _time_clock_to_frames(s, fps)
        return _time_code_to_frames(s, fps)
    return _atoi(s)


def to_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return _strtod(value)
