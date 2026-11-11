"""Fidelity harness: render synthetic Kdenlive projects with Kdenlive's own melt
(reference) and with this renderer, then compare frames.

    .venv/bin/python tests/fidelity.py [case ...]

Requires the Kdenlive Flatpak (for the reference melt) and the host FFmpeg.
"""
import os, subprocess, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, ".scratch", "fidelity")
os.makedirs(WORK, exist_ok=True)
W, H, FPS_N, FPS_D = 1280, 720, 24000, 1001
N = 48
MELT = ["flatpak", "run", "--env=LC_NUMERIC=C", "--env=LC_ALL=C.UTF-8", "--command=melt", "org.kde.kdenlive"]


def media():
    p = os.path.join(WORK, "src.mp4")
    if not os.path.exists(p):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"testsrc2=s={W}x{H}:r={FPS_N}/{FPS_D}:d=12",
                        "-f", "lavfi", "-i", "sine=f=440:d=12:sample_rate=48000",
                        "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", "-g", "24",
                        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                        "-c:a", "aac", "-shortest", p], check=True)
    return p


def media2():
    p = os.path.join(WORK, "src2.mp4")
    if not os.path.exists(p):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"smptehdbars=s=1920x1080:r=30:d=12,drawgrid=w=80:h=80:t=4:c=white",
                        "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", "-g", "30",
                        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", p],
                       check=True)
    return p


def filt(fid, service, **props):
    s = f'<filter id="{fid}"><property name="mlt_service">{service}</property>'
    s += f'<property name="kdenlive_id">{props.pop("kdenlive_id", service)}</property>'
    for k, v in props.items():
        s += f'<property name="{k.replace("__", ".")}">{v}</property>'
    return s + "</filter>"


def project(entries_v1, entries_v2=None, extra_transitions="", length=N, src=None):
    """entries: list of (in, out, filters_xml) or ('blank', n). Returns xml text."""
    src = src or media()
    def playlist(pid, entries, chain):
        out = f'<playlist id="{pid}">'
        for e in entries or []:
            if e[0] == "blank":
                out += f'<blank length="{e[1]}"/>'
            else:
                i, o, fx = e[0], e[1], e[2]
                ch = e[3] if len(e) > 3 else chain
                out += f'<entry in="{i}" out="{o}" producer="{ch}">{fx}</entry>'
        return out + "</playlist>"
    x = f'''<?xml version='1.0' encoding='utf-8'?>
<mlt LC_NUMERIC="C" producer="main_bin" root="{WORK}" version="7.41.0">
 <profile colorspace="709" description="test" display_aspect_den="9" display_aspect_num="16" frame_rate_den="{FPS_D}" frame_rate_num="{FPS_N}" height="{H}" progressive="1" sample_aspect_den="1" sample_aspect_num="1" width="{W}"/>
 <producer id="black" in="0" out="{length-1}"><property name="length">2147483647</property><property name="eof">continue</property><property name="resource">black</property><property name="mlt_service">color</property><property name="mlt_image_format">rgba</property><property name="set.test_audio">0</property></producer>
 <chain id="c1" out="287"><property name="length">288</property><property name="eof">pause</property><property name="resource">{src}</property><property name="mlt_service">avformat-novalidate</property><property name="seekable">1</property><property name="set.test_audio">1</property></chain>
 <chain id="c2" out="287"><property name="length">288</property><property name="eof">pause</property><property name="resource">{media2()}</property><property name="mlt_service">avformat-novalidate</property><property name="seekable">1</property><property name="set.test_audio">1</property></chain>
 {playlist("p1a", entries_v1, "c1")}<playlist id="p1b"/>
 <tractor id="t1"><track hide="audio" producer="p1a"/><track hide="audio" producer="p1b"/></tractor>
 {playlist("p2a", entries_v2, "c2")}<playlist id="p2b"/>
 <tractor id="t2"><track hide="audio" producer="p2a"/><track hide="audio" producer="p2b"/></tractor>
 <tractor id="seq" in="0" out="{length-1}">
  <track producer="black"/><track producer="t1"/><track producer="t2"/>
  <transition id="tr1"><property name="a_track">0</property><property name="b_track">1</property><property name="mlt_service">qtblend</property><property name="kdenlive_id">qtblend</property><property name="internal_added">237</property><property name="always_active">1</property><property name="compositing">0</property><property name="distort">0</property><property name="rotate_center">0</property></transition>
  <transition id="tr2"><property name="a_track">0</property><property name="b_track">2</property><property name="mlt_service">qtblend</property><property name="kdenlive_id">qtblend</property><property name="internal_added">237</property><property name="always_active">1</property><property name="compositing">0</property><property name="distort">0</property><property name="rotate_center">0</property></transition>
  {extra_transitions}
 </tractor>
 <tractor id="main" in="0" out="{length-1}"><property name="kdenlive:projectTractor">1</property><track in="0" out="{length-1}" producer="seq"/></tractor>
</mlt>'''
    return x


K = "00:00:00.000="
CASES = {
    "plain": lambda: project([(0, N - 1, "")]),
    "contrast0r": lambda: project([(0, N - 1, filt("f1", "frei0r.contrast0r", Contrast=K + "0.586", version="0.2"))]),
    "brightness_f0r": lambda: project([(0, N - 1, filt("f1", "frei0r.brightness", Brightness=K + "0.544", version="0.2"))]),
    "saturat0r": lambda: project([(0, N - 1, filt("f1", "frei0r.saturat0r", Saturation=K + "0.146", version="0.3"))]),
    "exposure": lambda: project([(0, N - 1, filt("f1", "avfilter.exposure", av__exposure=K + "0.329577", av__black=K + "0.046"))]),
    "colorlevels": lambda: project([(0, N - 1, filt("f1", "avfilter.colorlevels", av__gomax="0.95", av__rimax="1", av__gimax="1"))]),
    "fade_in": lambda: project([(0, N - 1, filt("f1", "brightness", kdenlive_id="fade_from_black", level="0=0;24=1", alpha="1", **{"in": "0", "out": "24"}).replace('<filter id="f1">', '<filter id="f1" in="0" out="24">'))]),
    "transform": lambda: project([(0, N - 1, filt("f1", "qtblend", rect="0=0 0 1280 720 1;47=320 180 640 360 0.5", rotation="0=0;47=30", rotate_center="1", compositing="0", distort="0", kdenlive_id="qtblend"))]),
    "pip_1080_src": lambda: project([(0, N - 1, "")], [(0, N - 1, filt("f2", "qtblend", rect="0 0 640 360 1", compositing="0", distort="0", kdenlive_id="qtblend"))]),
    "cut_and_blank": lambda: project([(30, 45, ""), ("blank", 8), (100, 123, "")]),
    "all5": lambda: project([(0, N - 1,
        filt("f1", "frei0r.contrast0r", Contrast=K + "0.586", version="0.2") +
        filt("f2", "avfilter.colorlevels", av__gomax="0.95") +
        filt("f3", "frei0r.brightness", Brightness=K + "0.544", version="0.2") +
        filt("f4", "avfilter.exposure", av__exposure=K + "0.329577", av__black=K + "0.046") +
        filt("f5", "frei0r.saturat0r", Saturation=K + "0.146", version="0.3"))]),
}

def s0t(**kw):
    """frei0r scale0tilt filter; kw uses short names (cl, cr, ct, cb, sx, sy, tx, ty)."""
    names = {"cl": "Clip left", "cr": "Clip right", "ct": "Clip top", "cb": "Clip bottom",
             "sx": "Scale X", "sy": "Scale Y", "tx": "Tilt X", "ty": "Tilt Y"}
    props = {names[k]: str(v) for k, v in kw.items()}
    return filt("f1", "frei0r.scale0tilt", kdenlive_id="frei0r.scale0tilt", **props)


CASES.update({
    "s0t_default": lambda: project([(0, N - 1, s0t(sx=0.5, sy=0.5, tx=0.5, ty=0.5))]),
    "s0t_zoom": lambda: project([(0, N - 1, s0t(sx=0.8, sy=0.8, tx=0.5, ty=0.5))]),
    "s0t_zoom_centre": lambda: project([(0, N - 1, s0t(sx=0.8, sy=0.8, tx=0.35, ty=0.35))]),
    "s0t_shrink": lambda: project([(0, N - 1, s0t(sx=0.25, sy=0.25, tx=0.75, ty=0.75))]),
    "s0t_crop": lambda: project([(0, N - 1, s0t(cl=0.1, cr=0.15, ct=0.2, cb=0.05, sx=0.5, sy=0.5, tx=0.5, ty=0.5))]),
    "s0t_anim": lambda: project([(0, N - 1, s0t(sx="00:00:00.000=0.5;00:00:01.960=0.8", sy="00:00:00.000=0.5;00:00:01.960=0.8",
                                                tx="00:00:00.000=0.5;00:00:01.960=0.35", ty="00:00:00.000=0.5;00:00:01.960=0.35"))]),
    "s0t_odd": lambda: project([(0, N - 1, s0t(cl=0.07, ct=0.03, sx=0.63, sy=0.41, tx=0.44, ty=0.58))]),
})


def host(cmd, **kw):
    return subprocess.run(cmd, **kw)


def load(p, n):
    """Frames as (n, 3, H, W) float arrays of the Y, U, V planes (4:4:4)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-frames:v", str(n), "-f", "rawvideo",
                          "-pix_fmt", "yuv444p", "-"], capture_output=True).stdout
    fs = W * H * 3
    k = len(raw) // fs
    return np.frombuffer(raw[: k * fs], np.uint8).reshape(k, 3, H, W).astype(np.float32)


def run_case(name):
    xml = CASES[name]()
    pj = os.path.join(WORK, f"{name}.kdenlive")
    open(pj, "w").write(xml)
    ref = os.path.join(WORK, f"{name}_ref.mkv")
    mine = os.path.join(WORK, f"{name}_mine.mkv")
    r = host(MELT + [pj, "-consumer", f"avformat:{ref}", "vcodec=ffv1", "pix_fmt=yuv444p", "an=1"],
             capture_output=True, text=True)
    if not os.path.exists(ref):
        print(name, "REF FAILED", r.stderr[-500:])
        return None
    # lossless 4:4:4 output from our renderer for a fair comparison
    env = dict(os.environ, KNR_LOSSLESS="1")
    r2 = host([sys.executable, "-m", "kdenlive_nvidia_renderer.cli", pj, mine, "--no-audio"],
              capture_output=True, text=True, cwd=ROOT, env=env)
    if r2.returncode != 0:
        print(name, "MINE FAILED", r2.stdout[-1500:], r2.stderr[-1500:])
        return None
    a, b = load(mine, N), load(ref, N)
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    mse = np.mean(d ** 2, axis=(2, 3))                  # (n, 3)
    psnr = 10 * np.log10(255 ** 2 / np.maximum(mse, 1e-6))
    y = psnr[:, 0]
    print(f"{name:16s} frames {len(a)}/{len(b)}  PSNR Y {y.mean():6.2f} (min {y.min():6.2f} @ {y.argmin()})"
          f"  U {psnr[:, 1].mean():6.2f}  V {psnr[:, 2].mean():6.2f}  maxdiff {np.abs(d).max():.0f}"
          f"  meanY {d[:, 0].mean():+.2f}")
    return psnr


if __name__ == "__main__":
    names = sys.argv[1:] or list(CASES)
    for nm in names:
        run_case(nm)
