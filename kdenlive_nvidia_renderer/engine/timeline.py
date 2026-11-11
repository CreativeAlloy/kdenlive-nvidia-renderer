"""Timeline evaluation: turn (project, frame) into a render plan.

The plan is a small tree of plain Python objects describing *what* has to be
drawn/played at one timeline position – which clip frames, which filters at
which filter-relative positions, and which transitions.  It contains no pixel
data, so it can be built cheaply ahead of time (e.g. to warm up decoders) and
consumed independently by the GPU video executor and the CPU audio mixer.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from ..mlt.project import Blank, Entry, Filter, Playlist, Producer, Project, Tractor, Transition


@dataclass
class FilterUse:
    filter: Filter
    position: int      # relative to filter "in"
    length: int


@dataclass
class TransitionUse:
    transition: Transition
    position: int      # relative to transition "in"
    length: int


@dataclass
class Node:
    key: tuple
    filters: list[FilterUse] = field(default_factory=list)
    no_video: bool = False
    no_audio: bool = False


@dataclass
class ClipNode(Node):
    producer: Producer | None = None
    frame: int = 0                 # producer frame position (timeline fps units)


@dataclass
class StackNode(Node):
    tractor: Tractor | None = None
    tracks: list[Node | None] = field(default_factory=list)
    hide_video: list[bool] = field(default_factory=list)
    hide_audio: list[bool] = field(default_factory=list)
    transitions: list[TransitionUse] = field(default_factory=list)


def _filter_active(f: Filter, pos: int) -> bool:
    if f.disabled:
        return False
    out = f.out or 0
    if f.in_ == 0 and out == 0:
        return True
    return pos >= f.in_ and (out == 0 or pos <= out)


def _uses(filters: list[Filter], pos: int, default_in: int, default_out: int) -> list[FilterUse]:
    res = []
    for f in filters:
        if not _filter_active(f, pos):
            continue
        out = f.out or 0
        if out > 0:
            length = out - f.in_ + 1
        else:
            length = default_out - default_in + 1
        res.append(FilterUse(f, pos - f.in_, max(1, length)))
    return res


class Timeline:
    def __init__(self, project: Project):
        self.project = project
        self._starts: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ helpers
    def _playlist_starts(self, pl: Playlist) -> list[int]:
        st = self._starts.get(pl.id)
        if st is None:
            st, acc = [], 0
            for it in pl.items:
                st.append(acc)
                acc += it.length
            self._starts[pl.id] = st
        return st

    def producer_range(self, prod: Producer) -> tuple[int, int]:
        if prod.out is not None:
            return prod.in_, prod.out
        ln = prod.length
        return prod.in_, max(prod.in_, (ln or 15000) - 1)

    @property
    def range(self) -> tuple[int, int]:
        main = self.project.main
        length = self.project.length_of(main.id)
        out = main.out if main.out is not None else main.in_ + length - 1
        return main.in_, out

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, pos: int) -> Node | None:
        return self._resolve(self.project.main.id, pos, ("root",))

    def _resolve(self, sid: str, pos: int, key: tuple) -> Node | None:
        svc = self.project.get(sid)
        if svc is None:
            return None
        if isinstance(svc, Producer):
            return self._resolve_producer(svc, pos, key)
        if isinstance(svc, Playlist):
            return self._resolve_playlist(svc, pos, key)
        if isinstance(svc, Tractor):
            return self._resolve_tractor(svc, pos, key)
        return None

    def _resolve_producer(self, prod: Producer, pos: int, key: tuple) -> Node | None:
        p_in, p_out = self.producer_range(prod)
        eof = prod.props.get("eof") or ""
        if eof == "continue" and pos > p_out and prod.service not in ("color", "colour"):
            return None
        node = ClipNode(key=key + (prod.id,), producer=prod, frame=pos)
        node.no_video = prod.props.get_bool("set.test_image")
        node.no_audio = prod.props.get_bool("set.test_audio")
        node.filters = _uses(prod.filters, pos, p_in, p_out)
        return node

    def _resolve_playlist(self, pl: Playlist, pos: int, key: tuple) -> Node | None:
        if pos < 0:
            return None
        starts = self._playlist_starts(pl)
        idx = bisect.bisect_right(starts, pos) - 1
        if idx < 0 or idx >= len(pl.items):
            return None
        item = pl.items[idx]
        if isinstance(item, Blank):
            return None
        local = pos - starts[idx]
        if local >= item.length:
            return None
        assert isinstance(item, Entry)
        src = item.in_ + local
        child = self._resolve(item.producer, src, key + (pl.id, idx))
        if child is None:
            return None
        # Entry filters see the parent producer position; default length is the parent's range.
        prod = self.project.get(item.producer)
        if isinstance(prod, Producer):
            d_in, d_out = self.producer_range(prod)
        else:
            d_in, d_out = 0, max(0, self.project.length_of(item.producer) - 1)
        extra = _uses(item.filters, src, d_in, d_out)
        if pl.filters:
            extra += _uses(pl.filters, pos, 0, max(0, pl.length - 1))
        if extra:
            child.filters = list(child.filters) + extra
        return child

    def _resolve_tractor(self, tr: Tractor, pos: int, key: tuple) -> Node | None:
        node = StackNode(key=key + (tr.id,), tractor=tr)
        any_child = False
        for i, t in enumerate(tr.tracks):
            tpos = pos + (t.in_ or 0)
            child = None
            if t.out is None or tpos <= t.out:
                child = self._resolve(t.producer, tpos, node.key + (i,))
            node.tracks.append(child)
            node.hide_video.append(t.hide_video)
            node.hide_audio.append(t.hide_audio)
            any_child = any_child or child is not None
        if not any_child:
            return None
        for tn in tr.transitions:
            if tn.disabled:
                continue
            out = tn.out or 0
            tractor_len = max(1, self.project.length_of(tr.id))
            if tn.props.get_bool("always_active"):
                tin = tn.in_
                length = out - tn.in_ + 1 if out > 0 else tractor_len
            elif tn.in_ == 0 and out == 0:
                tin, length = 0, tractor_len
            elif tn.in_ <= pos <= out:
                tin, length = tn.in_, out - tn.in_ + 1
            else:
                continue
            node.transitions.append(TransitionUse(tn, pos - tin, max(1, length)))
        tlen = self.project.length_of(tr.id)
        node.filters = _uses(tr.filters, pos, 0, max(0, tlen - 1))
        return node

    # ------------------------------------------------------------------ inventory
    def iter_clip_nodes(self, node: Node | None):
        if node is None:
            return
        if isinstance(node, ClipNode):
            yield node
        elif isinstance(node, StackNode):
            for c in node.tracks:
                yield from self.iter_clip_nodes(c)
