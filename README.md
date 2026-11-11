# Kdenlive Nvidia Renderer

Renders Kdenlive project files (`.kdenlive`) on an NVIDIA GPU instead of the CPU.

Kdenlive hands its projects to the MLT engine, which decodes, applies effects and composites every frame on the CPU. This application reads the same project XML and does all of that work on the GPU instead:

```
FFmpeg + NVDEC  ──►  CUDA kernels (CuPy)  ──►  FFmpeg + NVENC
 (decode, GPU       effects, keyframes,        (H.264 / HEVC / AV1)
  downscale)        transforms, compositing,
                    transitions
```

The audio is mixed on the CPU with numpy. That step takes well under a second.

Each effect is a port of the MLT, frei0r or FFmpeg code that Kdenlive uses, so the output matches what Kdenlive renders. This includes MLT's keyframe interpolation (all of its easing types) and its timecode rounding rules.

## Requirements

- Linux with an NVIDIA GPU and driver (tested: RTX 5090, driver 615, Linux Mint 22.3)
- `ffmpeg` / `ffprobe` with NVENC/NVDEC (Ubuntu/Mint's packaged FFmpeg works)
- Python 3.10+ with `venv`

## Running

```bash
./run.sh                      # first run creates .venv and installs PySide6, numpy, CuPy
./run.sh path/to/project.kdenlive
./install-desktop-entry.sh    # optional: app menu, "Open With", and a right-click "Render with Nvidia" in Nemo
```

There is also a command-line front end:

```bash
.venv/bin/python -m kdenlive_nvidia_renderer.cli project.kdenlive out.mp4 [--range 0:239] [--codec hevc_nvenc]
.venv/bin/python -m kdenlive_nvidia_renderer.cli project.kdenlive --analyze
```

## What is supported

| Area | Supported |
|---|---|
| Clips | video/audio files (`avformat`), images (`qimage`/`pixbuf`), color clips, speed changes (`timewarp`), nested sequences, rotated phone video |
| Timeline | tracks, blanks, track mute/hide, clip/track/master effects, effect in/out, keyframes (all MLT interpolation types) |
| Video effects | Brightness, Contrast, Saturation (frei0r), Exposure, Color Levels, Fade from/to black, Transform (`qtblend`), Crop/Scale/Tilt (frei0r `scale0tilt`), Horizontal/Vertical flip, Gaussian blur |
| Transitions | Composite & Transform (`qtblend`, all Qt blend modes), Composite / Wipe (`composite`), Dissolve / Wipe (`luma`, including luma files) |
| Audio | Volume / Gain / Fade in / Fade out / Mute, Pan / Balance, Mono, Channel copy, Deesser, audio crossfades and track mixing |
| Output | H.264 / HEVC / AV1 via NVENC, MP4 / MKV / MOV, AAC audio, custom resolution and frame rate, whole timeline or timeline zone |

The app lists every effect in the opened project and marks each one as **GPU** or
**Unsupported – skipped**. Unsupported items are left out of the render; they are
never approximated silently. Title clips (`kdenlivetitle`) are not supported yet.

## Fidelity

`tests/fidelity.py` builds synthetic Kdenlive projects and renders each one twice: once with Kdenlive's own `melt` (from the Flatpak) as the reference, and once with this renderer. It then compares the frames:

| Case | Result vs. Kdenlive (luma PSNR) |
|---|---|
| plain clip, cuts and blanks, fade from black | bit-exact |
| frei0r brightness / contrast / saturation, exposure, color levels | 46–54 dB |
| keyframed Transform with rotation and opacity | 44 dB |
| Crop, Scale and Tilt (crops, zooms, shrinks, keyframes) | 45–64 dB |
| Deesser audio | bit-exact |
| real project (`Romance.kdenlive`, 5 chained color effects) | 58 dB |

For reference, results above roughly 45 dB look the same to the eye. `tests/audio_fidelity.py` does the same for audio and compares the samples.

## Speed

On a 7-minute 720p project with 5 color effects, on an RTX 5090:

| Renderer | Time |
|---|---|
| Kdenlive / melt (CPU) | ~6 min (~28 fps) |
| this renderer, H.264 NVENC | **38 s** (~290 fps) |

The Deesser is the one audio effect that runs sample by sample, so it can't be vectorised; its state lives on the GPU. It costs about 2.6 ms per video frame.

## Note on Kdenlive and number locales

If your system uses a number locale with a decimal comma (for example `LC_NUMERIC=bg_BG.UTF-8`), Kdenlive's Flatpak build passes values such as `0.95` to FFmpeg filters, and FFmpeg can't parse them. Every `avfilter.*` effect with a decimal parameter, such as Exposure or Color Levels, is then dropped from the render without any error shown. This renderer applies those effects correctly. To fix it in Kdenlive itself:

```bash
flatpak override --user --env=LC_NUMERIC=C org.kde.kdenlive
```
