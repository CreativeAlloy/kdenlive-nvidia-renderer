"""Command-line front end (also used for testing).

    python -m kdenlive_nvidia_renderer.cli project.kdenlive out.mp4 [--range 0:240]
"""

from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction

from .engine.analyze import analyze
from .engine.encode import EncodeSettings
from .engine.render import render
from .mlt.project import load_project


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render a Kdenlive project on an NVIDIA GPU")
    ap.add_argument("project")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--range", help="timeline frames IN:OUT (inclusive)")
    ap.add_argument("--size", help="WxH output size (default: project profile)")
    ap.add_argument("--fps", help="output frame rate, e.g. 30 or 30000/1001")
    ap.add_argument("--codec", default="h264_nvenc", choices=["h264_nvenc", "hevc_nvenc", "av1_nvenc"])
    ap.add_argument("--preset", default="p5")
    ap.add_argument("--cq", type=int, default=19)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--analyze", action="store_true", help="only print the project analysis")
    a = ap.parse_args(argv)

    project = load_project(a.project)
    rep = analyze(project)
    print(f"Profile: {rep.profile}; {rep.duration_frames} frames; {rep.clips} clips")
    for it in rep.effects + rep.transitions + rep.producers:
        print(f"  [{it.status:11}] {it.name} ({it.service}) x{it.count}")
    for m in rep.missing_files:
        print(f"  MISSING: {m}")
    if a.analyze or not a.output:
        return 0

    w, h = project.profile.width, project.profile.height
    if a.size:
        w, h = (int(x) for x in a.size.lower().split("x"))
    fr = None
    if a.range:
        i, o = a.range.split(":")
        fr = (int(i), int(o))
    settings = EncodeSettings(output=a.output, width=w, height=h,
                              fps=Fraction(a.fps) if a.fps else project.profile.fps_fraction,
                              codec=a.codec, preset=a.preset, quality=a.cq, audio=not a.no_audio,
                              lossless_debug=bool(os.environ.get("KNR_LOSSLESS")))

    def prog(p):
        if p.stage == "video":
            eta = f"{p.eta:5.0f}s" if p.eta is not None else "  ?"
            print(f"\r  video {p.done}/{p.total}  {p.fps:6.1f} fps  ETA {eta}", end="", flush=True)
        elif p.stage == "audio" and p.done == 0:
            print("  mixing audio...", flush=True)

    elapsed = render(project, settings, fr, progress=prog, log=lambda m: print("\n  " + m))
    print(f"\nDone in {elapsed:.1f}s -> {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
