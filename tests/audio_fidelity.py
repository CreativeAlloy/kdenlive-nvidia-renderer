"""Audio fidelity harness: mix a synthetic project with Kdenlive's own melt (reference) and with this
renderer, then compare the samples.

    .venv/bin/python tests/audio_fidelity.py [case ...]

The source is a stereo mix of white-noise bursts (sibilance), pink noise and a tone.
Requires the Kdenlive Flatpak (for the reference melt) and the host FFmpeg.
"""
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import fidelity as f  # noqa: E402

FRAMES = 120
SRC = os.path.join(f.WORK, "sibilant.mkv")


def media():
    if not os.path.exists(SRC):
        subprocess.run([
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=s=320x180:r={f.FPS_N}/{f.FPS_D}:d=12",
            "-f", "lavfi", "-i", "anoisesrc=d=12:c=white:a=0.5:r=48000:seed=1",
            "-f", "lavfi", "-i", "anoisesrc=d=12:c=pink:a=0.5:r=48000:seed=2",
            "-f", "lavfi", "-i", "sine=f=220:d=12:sample_rate=48000",
            "-filter_complex",
            "[1]volume=enable='between(mod(t,1),0.2,0.45)':volume=1[a];[2][3]amix=inputs=2:normalize=0[b];"
            "[a][b]amerge=inputs=2,pan=stereo|c0=c0|c1=c1[aud]",
            "-map", "0:v", "-map", "[aud]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le", "-t", "12", SRC], check=True)
    return SRC


def deesser(**opts):
    props = {f"av__{k}": v for k, v in opts.items()}
    return f.filt("f1", "avfilter.deesser", kdenlive_id="avfilter.deesser", **props)


CASES = {
    "plain": "",
    "deesser": deesser(i="0.5", m="0.8", f="0.4", s="o"),
    "deesser_ess": deesser(i="0.4", m="0.9", f="0.3", s="e"),
    "deesser_kdenlive_defaults": deesser(i="0", m="0.8", f="0.01", s="o"),
    # Frequency 30 is outside FFmpeg's 0..1 range, so FFmpeg keeps its own default (0.5).
    "deesser_out_of_range": deesser(i="0.5", m="0.8", f="30", s="o"),
}


def project(name, fx):
    xml = (f.project([(0, FRAMES - 1, fx)], src=media(), length=FRAMES)
           .replace('hide="audio"', 'hide="video"')
           .replace('"set.test_audio">1<', '"set.test_audio">0<'))
    path = os.path.join(f.WORK, f"audio_{name}.kdenlive")
    with open(path, "w") as fh:
        fh.write(xml)
    return path


def reference(path, name):
    out = os.path.join(f.WORK, f"audio_{name}_ref.wav")
    subprocess.run(f.MELT + [path, "-consumer", f"avformat:{out}", "acodec=pcm_f32le", "vn=1",
                             "frequency=48000", "channels=2", "in=0", f"out={FRAMES - 1}"],
                   capture_output=True, text=True)
    if not os.path.exists(out):
        return None
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2)


def mine(path):
    from kdenlive_nvidia_renderer.engine.audio import AudioExecutor
    from kdenlive_nvidia_renderer.engine.context import RenderContext
    from kdenlive_nvidia_renderer.engine.timeline import Timeline
    from kdenlive_nvidia_renderer.mlt.project import load_project
    pr = load_project(path)
    tl, ex = Timeline(pr), AudioExecutor(RenderContext(pr, 1280, 720))
    t0 = time.time()
    out = np.concatenate([ex.render(tl.evaluate(i), i) for i in range(FRAMES)])
    return out, time.time() - t0


if __name__ == "__main__":
    for name in sys.argv[1:] or list(CASES):
        pj = project(name, CASES[name])
        ref = reference(pj, name)
        if ref is None:
            print(f"{name}: REFERENCE FAILED")
            continue
        out, secs = mine(pj)
        n = min(len(ref), len(out))
        diff = np.abs(ref[:n] - out[:n])
        rms = lambda a: float(np.sqrt(np.mean(a ** 2)))  # noqa: E731
        print(f"{name:28s} samples {len(ref)}/{len(out)}  max diff {diff.max():.2e}  "
              f"level {rms(ref[:n]):.4f}  rms diff {rms(ref[:n] - out[:n]):.2e}  ({secs:.2f}s)")
