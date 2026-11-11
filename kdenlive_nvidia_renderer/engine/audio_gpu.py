"""GPU kernels for audio effects that are sample-by-sample recurrences.

A recurrence cannot be vectorised, and a Python loop over ~20 million samples is far too slow, so the
state lives on the GPU and one thread per channel walks the samples of each frame in order.  The
module is compiled without ``--use_fast_math``: these filters must follow IEEE double arithmetic.
"""

from __future__ import annotations

import math

import cupy as cp
import numpy as np

_SRC = r"""
extern "C" {

// FFmpeg libavfilter/af_deesser.c, one thread per channel.
// state[ch*8 + ...]: s1 s2 s3 ratioA ratioB iirSampleA iirSampleB flip
__global__ void deesser(const float* in, float* out, int n, int nch, double* state,
                        double intensity, double maxdess, double iirAmount, int mode)
{
    int ch = threadIdx.x;
    if (ch >= nch) return;
    double* st = state + ch * 8;
    double s1 = st[0], s2 = st[1], s3 = st[2];
    double ratioA = st[3], ratioB = st[4], iirA = st[5], iirB = st[6];
    int flip = (int)st[7];
    for (int i = 0; i < n; ++i) {
        double src = (double)in[i * nch + ch];
        double sample = src;
        s3 = s2; s2 = s1; s1 = sample;
        double m1 = (s1 - s2) * ((s1 - s2) / 1.3);
        double m2 = (s2 - s3) * ((s1 - s2) / 1.3);
        double sense = (m1 - m2) * ((m1 - m2) / 1.3);
        double attackspeed = 7.0 + sense * 1024;

        sense = 1.0 + intensity * intensity * sense;
        sense = (sense > intensity) ? intensity : sense;          // FFMIN(sense, intensity)
        double recovery = 1.0 + (0.01 / sense);
        double offset = 1.0 - fabs(sample);

        if (flip) {
            iirA = (iirA * (1.0 - (offset * iirAmount))) + (sample * (offset * iirAmount));
            if (ratioA < sense) ratioA = ((ratioA * attackspeed) + sense) / (attackspeed + 1.0);
            else                ratioA = 1.0 + ((ratioA - 1.0) / recovery);
            ratioA = (ratioA > maxdess) ? maxdess : ratioA;       // FFMIN(ratioA, maxdess)
            sample = iirA + ((sample - iirA) / ratioA);
        } else {
            iirB = (iirB * (1.0 - (offset * iirAmount))) + (sample * (offset * iirAmount));
            if (ratioB < sense) ratioB = ((ratioB * attackspeed) + sense) / (attackspeed + 1.0);
            else                ratioB = 1.0 + ((ratioB - 1.0) / recovery);
            ratioB = (ratioB > maxdess) ? maxdess : ratioB;
            sample = iirB + ((sample - iirB) / ratioB);
        }
        flip = !flip;

        double o = (mode == 0) ? src : (mode == 1) ? sample : src - sample;   // input / output / ess
        out[i * nch + ch] = (float)o;
    }
    st[0] = s1; st[1] = s2; st[2] = s3;
    st[3] = ratioA; st[4] = ratioB; st[5] = iirA; st[6] = iirB; st[7] = (double)flip;
}

}  // extern "C"
"""

_module: cp.RawModule | None = None


def _k(name: str):
    global _module
    if _module is None:
        _module = cp.RawModule(code=_SRC)
    return _module.get_function(name)


def new_deesser_state(channels: int) -> cp.ndarray:
    """Fresh state: ratioA = ratioB = 1.0 (config_input() of the filter), everything else zero."""
    st = np.zeros((channels, 8), dtype=np.float64)
    st[:, 3] = 1.0
    st[:, 4] = 1.0
    return cp.asarray(st)


def deesser(samples: np.ndarray, state: cp.ndarray, intensity: float, max_amount: float,
            frequency: float, mode: int, sample_rate: int = 48000) -> np.ndarray:
    """De-ess one block of interleaved float32 samples (n, channels); ``state`` is updated in place.

    ``mode``: 0 input (unchanged), 1 output (default), 2 ess (the removed part only).
    """
    n, ch = samples.shape
    if n == 0:
        return samples
    # constants exactly as computed per block in filter_frame()
    overallscale = 44100.0 / sample_rate if sample_rate < 44100 else sample_rate / 44100.0
    inten = math.pow(intensity, 5) * (8192 / overallscale)
    maxdess = 1.0 / math.pow(10.0, ((max_amount - 1.0) * 48.0) / 20)
    iir_amount = math.pow(frequency, 2) / overallscale
    src = cp.asarray(np.ascontiguousarray(samples, dtype=np.float32))
    out = cp.empty_like(src)
    _k("deesser")((1,), (ch,), (src, out, np.int32(n), np.int32(ch), state, np.float64(inten),
                                np.float64(maxdess), np.float64(iir_amount), np.int32(mode)))
    return cp.asnumpy(out)
