"""Shared render context: output geometry, warnings, luma-map cache."""

from __future__ import annotations

import glob
import os
from fractions import Fraction

from ..mlt.project import Project

LUMA_DIRS = [
    "/usr/share/kdenlive/lumas", "/usr/local/share/kdenlive/lumas",
    os.path.expanduser("~/.local/share/kdenlive/lumas"),
    "/var/lib/flatpak/app/org.kde.kdenlive/current/active/files/share/kdenlive/lumas",
    os.path.expanduser("~/.local/share/flatpak/app/org.kde.kdenlive/current/active/files/share/kdenlive/lumas"),
    "/usr/share/mlt-7/lumas", "/usr/share/mlt/lumas",
    "/var/lib/flatpak/app/org.kde.kdenlive/current/active/files/share/mlt-7/lumas",
]


class RenderContext:
    def __init__(self, project: Project, out_w: int, out_h: int, log=None):
        p = project.profile
        self.project = project
        self.profile_w = p.width
        self.profile_h = p.height
        self.profile_sar = p.sar or 1.0
        self.frame_dar = p.dar
        self.out_w = out_w
        self.out_h = out_h
        self.sx = out_w / p.width
        self.sy = out_h / p.height
        self.fps = p.fps
        self.fps_fraction = Fraction(p.fps_num, p.fps_den)
        self.colorspace = p.colorspace if p.colorspace in (601, 709, 2020) else (709 if p.height >= 720 else 601)
        self._log = log or (lambda *_: None)
        self._warned: set[str] = set()
        self._luma_cache: dict = {}

    def log(self, msg: str):
        self._log(msg)

    def warn(self, msg: str):
        if msg not in self._warned:
            self._warned.add(msg)
            self._log("WARNING: " + msg)

    # ------------------------------------------------------------------ luma wipes
    def find_luma(self, resource: str) -> str | None:
        if not resource:
            return None
        res = resource.lstrip("%")
        if os.path.isabs(res) and os.path.exists(res):
            return res
        cand = os.path.join(self.project.root_dir, res)
        if os.path.exists(cand):
            return cand
        base = os.path.basename(res)
        hd = "HD" if self.profile_h >= 720 else "PAL"
        for d in LUMA_DIRS:
            for sub in (hd, "HD", "PAL", "NTSC", "square", ""):
                c = os.path.join(d, sub, base)
                if os.path.exists(c):
                    return c
            hits = glob.glob(os.path.join(d, "**", base), recursive=True)
            if hits:
                return hits[0]
        return None

    def luma_map(self, resource: str, invert: bool):
        """Luma map resized to the output frame as a float32 (H*W) GPU array in 0..1."""
        key = (resource, invert)
        if key in self._luma_cache:
            return self._luma_cache[key]
        path = self.find_luma(resource)
        result = None
        if path is None:
            self.warn(f"Luma wipe file not found: {resource} (falling back to a dissolve)")
        else:
            import subprocess
            import numpy as np
            import cupy as cp
            from .media import FFMPEG
            cmd = [FFMPEG, "-nostdin", "-loglevel", "error", "-i", path, "-frames:v", "1",
                   "-vf", f"scale={self.out_w}:{self.out_h}:flags=bicubic", "-f", "rawvideo",
                   "-pix_fmt", "gray16le", "pipe:1"]
            res = subprocess.run(cmd, capture_output=True)
            n = self.out_w * self.out_h
            if res.returncode == 0 and len(res.stdout) >= n * 2:
                arr = np.frombuffer(res.stdout[: n * 2], dtype=np.uint16).astype(np.float32) / 65535.0
                if invert:
                    arr = 1.0 - arr
                result = cp.asarray(arr)
            else:
                self.warn(f"Could not read luma wipe {path}")
        self._luma_cache[key] = result      # None => callers fall back to a plain dissolve
        return result
