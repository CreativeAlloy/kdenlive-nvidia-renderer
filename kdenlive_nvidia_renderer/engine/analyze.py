"""Static analysis of a project: what is in it and what this renderer supports.

Does not need the GPU stack, so the GUI can show it instantly.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field

from ..mlt.project import Entry, Playlist, Producer, Project, Tractor
from .effects import AUDIO_FILTERS, AUDIO_TRANSITIONS, IGNORED, VIDEO_FILTERS, VIDEO_TRANSITIONS

SUPPORTED_PRODUCERS = {"avformat", "avformat-novalidate", "timewarp", "qimage", "pixbuf", "color",
                       "colour"}


@dataclass
class Item:
    name: str
    service: str
    count: int
    status: str        # "gpu", "audio", "ignored", "unsupported"


@dataclass
class Report:
    profile: str
    duration_frames: int
    fps: float
    tracks_video: int
    tracks_audio: int
    clips: int
    effects: list[Item] = field(default_factory=list)
    transitions: list[Item] = field(default_factory=list)
    producers: list[Item] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    zone: tuple[int, int] | None = None

    @property
    def unsupported(self) -> list[Item]:
        return [i for i in self.effects + self.transitions + self.producers if i.status == "unsupported"]


def _reachable(project: Project) -> set[str]:
    seen: set[str] = set()
    stack = [project.main.id]
    while stack:
        sid = stack.pop()
        if sid in seen:
            continue
        seen.add(sid)
        svc = project.get(sid)
        if isinstance(svc, Tractor):
            stack.extend(t.producer for t in svc.tracks)
        elif isinstance(svc, Playlist):
            stack.extend(it.producer for it in svc.items if isinstance(it, Entry))
    return seen


def analyze(project: Project) -> Report:
    reach = _reachable(project)
    effects: Counter = Counter()
    eff_names: dict[str, str] = {}
    transitions: Counter = Counter()
    tr_names: dict[str, str] = {}
    producers: Counter = Counter()
    missing: set[str] = set()
    clips = 0

    def add_filters(filters):
        for f in filters:
            if f.disabled:
                continue
            effects[f.service] += 1
            eff_names.setdefault(f.service, f.kdenlive_id)

    for sid in reach:
        svc = project.get(sid)
        if isinstance(svc, Producer):
            add_filters(svc.filters)
        elif isinstance(svc, Playlist):
            add_filters(svc.filters)
            for it in svc.items:
                if isinstance(it, Entry):
                    clips += 1
                    add_filters(it.filters)
                    p = project.get(it.producer)
                    if isinstance(p, Producer):
                        producers[p.service] += 1
                        if p.resource_path and p.service in SUPPORTED_PRODUCERS - {"color", "colour"}:
                            if not os.path.exists(p.resource_path):
                                missing.add(p.resource_path)
        elif isinstance(svc, Tractor):
            add_filters(svc.filters)
            for t in svc.transitions:
                if t.disabled:
                    continue
                transitions[t.service] += 1
                tr_names.setdefault(t.service, t.kdenlive_id)

    def f_status(s):
        if s in IGNORED:
            return "ignored"
        if s in VIDEO_FILTERS:
            return "gpu"
        if s in AUDIO_FILTERS:
            return "audio"
        return "unsupported"

    def t_status(s):
        if s in VIDEO_TRANSITIONS:
            return "gpu"
        if s in AUDIO_TRANSITIONS:
            return "audio"
        return "unsupported"

    # Count tracks of the active sequence (the main tractor's first track if it wraps one)
    main = project.main
    seq = main
    if len(main.tracks) == 1 and isinstance(project.get(main.tracks[0].producer), Tractor):
        seq = project.get(main.tracks[0].producer)
    tv = ta = 0
    for t in seq.tracks:
        sub = project.get(t.producer)
        if isinstance(sub, Tractor) and sub.props.get("kdenlive:audio_track") == "1":
            ta += 1
        elif isinstance(sub, Tractor):
            tv += 1

    zone = None
    zin = seq.props.get("kdenlive:sequenceproperties.zonein")
    zout = seq.props.get("kdenlive:sequenceproperties.zoneout")
    if zin is not None and zout is not None:
        try:
            zi, zo = int(zin), int(zout)
            if zo > zi:
                zone = (zi, zo)
        except ValueError:
            pass

    length = project.length_of(main.id)
    p = project.profile
    return Report(
        profile=f"{p.width}x{p.height} @ {p.fps:.3f} fps" + (f" ({p.description})" if p.description else ""),
        duration_frames=length, fps=p.fps, tracks_video=tv, tracks_audio=ta, clips=clips,
        effects=[Item(eff_names[s], s, n, f_status(s)) for s, n in effects.most_common()],
        transitions=[Item(tr_names[s], s, n, t_status(s)) for s, n in transitions.most_common()],
        producers=[Item(s, s, n, "gpu" if s in SUPPORTED_PRODUCERS else "unsupported")
                   for s, n in producers.most_common()],
        missing_files=sorted(missing), zone=zone)
