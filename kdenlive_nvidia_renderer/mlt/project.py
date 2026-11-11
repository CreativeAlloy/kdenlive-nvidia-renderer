"""Parse a .kdenlive (MLT XML) document into a lightweight object model.

The model mirrors MLT's own structure rather than Kdenlive's UI concepts so
that rendering semantics can follow MLT closely:

* ``Producer`` – a media source (avformat chain, colour, image, nested tractor…)
* ``Playlist`` – a sequence of ``Entry`` (clip cut) and ``Blank`` items
* ``Tractor``  – a stack of ``Track`` s plus ``Transition`` s and ``Filter`` s
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from fractions import Fraction

from .timecode import to_frames, to_float


@dataclass
class Profile:
    width: int = 1920
    height: int = 1080
    fps_num: int = 25
    fps_den: int = 1
    sar_num: int = 1
    sar_den: int = 1
    dar_num: int = 16
    dar_den: int = 9
    colorspace: int = 709
    progressive: bool = True
    description: str = ""

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    @property
    def fps_fraction(self) -> Fraction:
        return Fraction(self.fps_num, self.fps_den)

    @property
    def sar(self) -> float:
        return self.sar_num / self.sar_den if self.sar_den else 1.0

    @property
    def dar(self) -> float:
        return self.dar_num / self.dar_den if self.dar_den else self.width / self.height


class Props(dict):
    """Property bag with typed helpers. Missing keys yield ``None``."""

    def get_int(self, name, fps=0.0, default=0):
        v = self.get(name)
        return default if v is None or v == "" else to_frames(v, fps, default)

    def get_float(self, name, default=0.0):
        v = self.get(name)
        return default if v is None or v == "" else to_float(v, default)

    def get_bool(self, name, default=False):
        v = self.get(name)
        if v is None or v == "":
            return default
        return to_float(v) != 0


@dataclass
class Service:
    id: str
    props: Props
    filters: list["Filter"] = field(default_factory=list)
    in_: int = 0
    out: int | None = None

    @property
    def service(self) -> str:
        return self.props.get("mlt_service") or ""


@dataclass
class Filter(Service):
    @property
    def kdenlive_id(self) -> str:
        return self.props.get("kdenlive_id") or self.service

    @property
    def disabled(self) -> bool:
        return self.props.get_bool("disable")


@dataclass
class Transition(Service):
    a_track: int = 0
    b_track: int = 1

    @property
    def kdenlive_id(self) -> str:
        return self.props.get("kdenlive_id") or self.service

    @property
    def disabled(self) -> bool:
        return self.props.get_bool("disable")


@dataclass
class Link(Service):
    pass


@dataclass
class Producer(Service):
    tag: str = "producer"
    links: list[Link] = field(default_factory=list)
    resource_path: str | None = None     # resolved absolute path for file-based producers

    @property
    def resource(self) -> str:
        return self.props.get("resource") or ""

    @property
    def length(self) -> int:
        return self.props.get_int("length", default=0)


@dataclass
class Entry:
    producer: str
    in_: int
    out: int
    props: Props
    filters: list[Filter] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.out - self.in_ + 1


@dataclass
class Blank:
    length: int


@dataclass
class Playlist(Service):
    items: list = field(default_factory=list)

    @property
    def length(self) -> int:
        return sum(it.length for it in self.items)


@dataclass
class Track:
    producer: str
    hide_video: bool = False
    hide_audio: bool = False
    in_: int | None = None
    out: int | None = None


@dataclass
class Tractor(Service):
    tracks: list[Track] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)


@dataclass
class Project:
    path: str
    root_dir: str
    profile: Profile
    services: dict[str, Service]
    main: Tractor
    bin_props: Props
    warnings: list[str] = field(default_factory=list)

    def get(self, sid: str) -> Service | None:
        return self.services.get(sid)

    def length_of(self, sid: str) -> int:
        """Length (frames) of any service, as MLT would report it."""
        svc = self.services.get(sid)
        if svc is None:
            return 0
        if isinstance(svc, Playlist):
            return svc.length
        if isinstance(svc, Tractor):
            if svc.out is not None:
                return svc.out - svc.in_ + 1
            return max((self.track_length(t) for t in svc.tracks), default=0)
        if isinstance(svc, Producer):
            if svc.out is not None:
                return svc.out + 1
            return svc.length
        return 0

    def track_length(self, t: Track) -> int:
        if t.out is not None:
            return t.out - (t.in_ or 0) + 1
        return self.length_of(t.producer)

    @property
    def fps(self) -> float:
        return self.profile.fps


# ----------------------------------------------------------------------------

def _props(el: ET.Element) -> Props:
    p = Props()
    for prop in el.findall("property"):
        name = prop.get("name")
        if name is None:
            continue
        # Properties can contain nested XML (e.g. titles); keep raw text for those.
        if len(prop):
            p[name] = "".join(ET.tostring(c, encoding="unicode") for c in prop)
        else:
            p[name] = prop.text or ""
    return p


def _opt_frames(v, fps):
    return None if v is None or v == "" else to_frames(v, fps)


def _parse_filter(el: ET.Element, fps: float) -> Filter:
    return Filter(id=el.get("id", ""), props=_props(el),
                  in_=to_frames(el.get("in"), fps), out=_opt_frames(el.get("out"), fps))


def load_project(path: str) -> Project:
    path = os.path.abspath(path)
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "mlt":
        raise ValueError("Not an MLT/Kdenlive project (root element is <%s>)" % root.tag)

    prof_el = root.find("profile")
    profile = Profile()
    if prof_el is not None:
        g = prof_el.get
        profile = Profile(
            width=int(g("width", 1920)), height=int(g("height", 1080)),
            fps_num=int(g("frame_rate_num", 25)), fps_den=int(g("frame_rate_den", 1)),
            sar_num=int(g("sample_aspect_num", 1)), sar_den=int(g("sample_aspect_den", 1)),
            dar_num=int(g("display_aspect_num", 16)), dar_den=int(g("display_aspect_den", 9)),
            colorspace=int(g("colorspace", 709)), progressive=g("progressive", "1") == "1",
            description=g("description", ""))
    fps = profile.fps

    project_dir = os.path.dirname(path)
    root_dir = root.get("root") or project_dir
    services: dict[str, Service] = {}
    warnings: list[str] = []
    tractors: list[Tractor] = []
    bin_props = Props()

    for el in root:
        tag = el.tag
        sid = el.get("id", "")
        if tag in ("producer", "chain"):
            props = _props(el)
            prod = Producer(id=sid, props=props, tag=tag,
                            in_=to_frames(el.get("in"), fps), out=_opt_frames(el.get("out"), fps),
                            filters=[_parse_filter(f, fps) for f in el.findall("filter")],
                            links=[Link(id=l.get("id", ""), props=_props(l)) for l in el.findall("link")])
            prod.resource_path = _resolve_resource(prod, root_dir, project_dir)
            services[sid] = prod
        elif tag == "playlist":
            pl = Playlist(id=sid, props=_props(el),
                          filters=[_parse_filter(f, fps) for f in el.findall("filter")])
            for item in el:
                if item.tag == "entry":
                    pl.items.append(Entry(
                        producer=item.get("producer", ""),
                        in_=to_frames(item.get("in"), fps),
                        out=to_frames(item.get("out"), fps),
                        props=_props(item),
                        filters=[_parse_filter(f, fps) for f in item.findall("filter")]))
                elif item.tag == "blank":
                    pl.items.append(Blank(length=to_frames(item.get("length"), fps)))
            services[sid] = pl
            if sid == "main_bin":
                bin_props = pl.props
        elif tag == "tractor":
            tr = Tractor(id=sid, props=_props(el),
                         in_=to_frames(el.get("in"), fps), out=_opt_frames(el.get("out"), fps),
                         filters=[_parse_filter(f, fps) for f in el.findall("filter")])
            # MLT also allows <multitrack><track/></multitrack>
            track_els = el.findall("track") + el.findall("multitrack/track")
            for t in track_els:
                hide = t.get("hide", "")
                tr.tracks.append(Track(producer=t.get("producer", ""),
                                       hide_video=hide in ("video", "both"),
                                       hide_audio=hide in ("audio", "both"),
                                       in_=_opt_frames(t.get("in"), fps),
                                       out=_opt_frames(t.get("out"), fps)))
            for t in el.findall("transition"):
                p = _props(t)
                tr.transitions.append(Transition(
                    id=t.get("id", ""), props=p,
                    in_=to_frames(t.get("in") or p.get("in"), fps),
                    out=_opt_frames(t.get("out") or p.get("out"), fps),
                    a_track=p.get_int("a_track", default=0),
                    b_track=p.get_int("b_track", default=1)))
            services[sid] = tr
            tractors.append(tr)

    if not tractors:
        raise ValueError("Project contains no timeline (no <tractor> element)")

    main = next((t for t in tractors if t.props.get_bool("kdenlive:projectTractor")), tractors[-1])
    return Project(path=path, root_dir=root_dir, profile=profile, services=services,
                   main=main, bin_props=bin_props, warnings=warnings)


_FILE_SERVICES_PREFIX = ("avformat", "qimage", "pixbuf", "timewarp", "kdenlivetitle", "glaxnimate",
                         "xml", "consumer")


def _resolve_resource(prod: Producer, root_dir: str, project_dir: str) -> str | None:
    svc = prod.service
    res = prod.resource
    if not res or svc in ("color", "colour", "blipflash", "noise", "tone"):
        return None
    if svc == "timewarp" and ":" in res:
        # "speed:path"
        res = res.split(":", 1)[1]
    if svc == "kdenlivetitle" and not res.endswith(".kdenlivetitle"):
        return None
    if os.path.isabs(res):
        cand = [res]
    else:
        cand = [os.path.join(root_dir, res), os.path.join(project_dir, res)]
    for c in cand:
        if os.path.exists(c):
            return os.path.normpath(c)
        # image sequences / pattern resources
        if "%" in os.path.basename(c) or ".all." in os.path.basename(c):
            if os.path.isdir(os.path.dirname(c)):
                return os.path.normpath(c)
    return os.path.normpath(cand[0])
