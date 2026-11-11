"""CUDA kernels (via CuPy) for every per-pixel operation in the pipeline.

Frames live on the GPU as float32 ``(H, W, 4)`` RGBA arrays with *straight*
(non-premultiplied) alpha in the range 0..1, matching MLT's rgba image format.
Effects that are 8-bit in MLT (frei0r, avfilter on rgba) quantise their input
to 0..255 internally so results match Kdenlive's output exactly.
"""

from __future__ import annotations

import math
import os

import cupy as cp
import numpy as np

_SRC = r"""
extern "C" {

__device__ __forceinline__ float clamp01(float v) { return fminf(fmaxf(v, 0.f), 1.f); }
__device__ __forceinline__ int q8(float v) { return __float2int_rn(clamp01(v) * 255.f); }

// ------------------------------------------------------------------ input
__device__ __forceinline__ float cubic_w(float t, float a)
{
    t = fabsf(t);
    if (t <= 1.f) return ((a + 2.f) * t - (a + 3.f)) * t * t + 1.f;
    if (t < 2.f) return ((a * t - 5.f * a) * t + 8.f * a) * t - 4.f * a;
    return 0.f;
}

__device__ float chroma_bicubic(const unsigned char* uv8, const unsigned short* uv16, int is16,
                                int pitch, int cw, int ch, float cx, float cy, int comp, float a)
{
    int x0 = (int)floorf(cx), y0 = (int)floorf(cy);
    float fx = cx - x0, fy = cy - y0;
    float acc = 0.f, wsum = 0.f;
    for (int j = -1; j <= 2; ++j) {
        float wy = cubic_w(j - fy, a);
        int yy = min(max(y0 + j, 0), ch - 1);
        for (int i = -1; i <= 2; ++i) {
            float w = wy * cubic_w(i - fx, a);
            int xx = min(max(x0 + i, 0), cw - 1);
            float v = is16 ? uv16[yy * pitch + 2 * xx + comp] * (1.f / 65535.f)
                           : uv8[yy * pitch + 2 * xx + comp] * (1.f / 255.f);
            acc += w * v; wsum += w;
        }
    }
    return acc / wsum;
}
// NV12 / P010 (bit depth via 'scale': 1/255 or 1/65535 with 16-bit containers)
__global__ void yuv420_to_rgba(const unsigned char* y8, const unsigned char* uv8,
                               const unsigned short* y16, const unsigned short* uv16,
                               int w, int h, int y_pitch, int uv_pitch,
                               float kr, float kb, int full_range, int is16,
                               float* out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || yy >= h) return;
    float Y, U, V;
    // chroma reconstruction: 0 = nearest (2x2 replicate), 1 = bilinear left-sited (MPEG-2),
    // 2 = bilinear centre-sited (MPEG-1/JPEG)
    float cx, cy;
    if (CHROMA_MODE == 0) { cx = (float)(x / 2); cy = (float)(yy / 2); }
    else if (CHROMA_MODE == 1) { cx = x * 0.5f; cy = (yy + 0.5f) * 0.5f - 0.5f; }
    else { cx = (x + 0.5f) * 0.5f - 0.5f; cy = (yy + 0.5f) * 0.5f - 0.5f; }
    int cw = (w + 1) / 2, ch = (h + 1) / 2;
    int x0 = (int)floorf(cx), y0 = (int)floorf(cy);
    float fx = cx - x0, fy = cy - y0;
    int x1 = min(x0 + 1, cw - 1), y1 = min(y0 + 1, ch - 1);
    x0 = max(0, min(x0, cw - 1)); y0 = max(0, min(y0, ch - 1));
    float u00, u01, u10, u11, v00, v01, v10, v11;
    if (is16) {
        const float s = 1.f / 65535.f;
        Y = y16[yy * y_pitch + x] * s;
        u00 = uv16[y0 * uv_pitch + 2*x0] * s; v00 = uv16[y0 * uv_pitch + 2*x0 + 1] * s;
        u01 = uv16[y0 * uv_pitch + 2*x1] * s; v01 = uv16[y0 * uv_pitch + 2*x1 + 1] * s;
        u10 = uv16[y1 * uv_pitch + 2*x0] * s; v10 = uv16[y1 * uv_pitch + 2*x0 + 1] * s;
        u11 = uv16[y1 * uv_pitch + 2*x1] * s; v11 = uv16[y1 * uv_pitch + 2*x1 + 1] * s;
    } else {
        const float s = 1.f / 255.f;
        Y = y8[yy * y_pitch + x] * s;
        u00 = uv8[y0 * uv_pitch + 2*x0] * s; v00 = uv8[y0 * uv_pitch + 2*x0 + 1] * s;
        u01 = uv8[y0 * uv_pitch + 2*x1] * s; v01 = uv8[y0 * uv_pitch + 2*x1 + 1] * s;
        u10 = uv8[y1 * uv_pitch + 2*x0] * s; v10 = uv8[y1 * uv_pitch + 2*x0 + 1] * s;
        u11 = uv8[y1 * uv_pitch + 2*x1] * s; v11 = uv8[y1 * uv_pitch + 2*x1 + 1] * s;
    }
    U = (u00 * (1-fx) + u01 * fx) * (1-fy) + (u10 * (1-fx) + u11 * fx) * fy;
    V = (v00 * (1-fx) + v01 * fx) * (1-fy) + (v10 * (1-fx) + v11 * fx) * fy;
    if (CHROMA_MODE >= 3) {
        float a = (CHROMA_MODE == 3) ? -0.5f : -0.6f;
        float ccx = (CHROMA_MODE == 5) ? x * 0.5f : (x + 0.5f) * 0.5f - 0.5f;
        float ccy = (yy + 0.5f) * 0.5f - 0.5f;
        U = chroma_bicubic(uv8, uv16, is16, uv_pitch, cw, ch, ccx, ccy, 0, a);
        V = chroma_bicubic(uv8, uv16, is16, uv_pitch, cw, ch, ccx, ccy, 1, a);
    }
    float yn, cb, cr;
    if (full_range) { yn = Y; cb = U - 0.5f; cr = V - 0.5f; }
    else { yn = (Y - 16.f/255.f) * (255.f/219.f); cb = (U - 128.f/255.f) * (255.f/224.f);
           cr = (V - 128.f/255.f) * (255.f/224.f); }
    float kg = 1.f - kr - kb;
    float r = yn + 2.f * (1.f - kr) * cr;
    float b = yn + 2.f * (1.f - kb) * cb;
    float g = (yn - kr * r - kb * b) / kg;
    float* o = out + 4 * (yy * w + x);
    // Unclamped: a YUV->RGB->YUV round trip stays (near) lossless, like MLT's YUV passthrough.
    o[0] = r; o[1] = g; o[2] = b; o[3] = 1.f;
}

__global__ void rgba8_to_rgba(const unsigned char* in, int n, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 4) return;
    out[i] = in[i] * (1.f / 255.f);
}

// ------------------------------------------------------------------ output
// Flatten over black and convert to NV12 (limited range). One thread per 2x2 block.
__device__ __forceinline__ unsigned char code8(float v, float lo, float hi)
{
    return (unsigned char)__float2int_rn(fminf(fmaxf(v, lo), hi));
}

__global__ void rgba_to_nv12(const float* in, int w, int h, float kr, float kb,
                             unsigned char* y_out, unsigned char* uv_out)
{
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int cw = w / 2, ch = h / 2;
    if (bx >= cw || by >= ch) return;
    float kg = 1.f - kr - kb;
    float rs = 0, gs = 0, bs = 0;
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            int x = 2 * bx + dx, y = 2 * by + dy;
            const float* p = in + 4 * (y * w + x);
            float a = clamp01(p[3]);
            float r = p[0] * a, g = p[1] * a, b = p[2] * a;
            y_out[y * w + x] = code8(16.f + 219.f * (kr * r + kg * g + kb * b), 0.f, 255.f);
            rs += r; gs += g; bs += b;
        }
    rs *= 0.25f; gs *= 0.25f; bs *= 0.25f;
    float Y = kr * rs + kg * gs + kb * bs;
    float cb = (bs - Y) / (2.f * (1.f - kb));
    float cr = (rs - Y) / (2.f * (1.f - kr));
    uv_out[by * w + 2 * bx]     = code8(128.f + 224.f * cb, 0.f, 255.f);
    uv_out[by * w + 2 * bx + 1] = code8(128.f + 224.f * cr, 0.f, 255.f);
}

// Planar YUV 4:4:4 (limited range) for lossless fidelity testing
__global__ void rgba_to_yuv444(const float* in, int w, int h, float kr, float kb, unsigned char* out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float kg = 1.f - kr - kb;
    const float* p = in + 4 * (y * w + x);
    float a = clamp01(p[3]);
    float r = p[0] * a, g = p[1] * a, b = p[2] * a;
    float Y = kr * r + kg * g + kb * b;
    int i = y * w + x, n = w * h;
    out[i] = code8(16.f + 219.f * Y, 0.f, 255.f);
    out[n + i] = code8(128.f + 224.f * (b - Y) / (2.f * (1.f - kb)), 0.f, 255.f);
    out[2 * n + i] = code8(128.f + 224.f * (r - Y) / (2.f * (1.f - kr)), 0.f, 255.f);
}

// ------------------------------------------------------------------ 8-bit effects
// Per-channel 256-entry LUTs (R, G, B consecutively); alpha untouched.
__global__ void apply_lut3(float* img, int n, const unsigned char* lut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* p = img + 4 * i;
    p[0] = lut[q8(p[0])] * (1.f / 255.f);
    p[1] = lut[256 + q8(p[1])] * (1.f / 255.f);
    p[2] = lut[512 + q8(p[2])] * (1.f / 255.f);
}

__global__ void apply_alpha_lut(float* img, int n, const unsigned char* lut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* p = img + 4 * i;
    p[3] = lut[q8(p[3])] * (1.f / 255.f);
}

// frei0r saturat0r (weights applied to bytes in memory order, as upstream does)
__global__ void saturat0r(float* img, int n, int w0, int w1, int w2, float sat, int clampit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* p = img + 4 * i;
    int c0 = q8(p[0]), c1 = q8(p[1]), c2 = q8(p[2]);
    int bw = (c0 * w0 + c1 * w1 + c2 * w2) >> 16;
    float o0 = bw + c0 * sat, o1 = bw + c1 * sat, o2 = bw + c2 * sat;
    int r0, r1, r2;
    if (clampit) {
        r0 = max(0, min(255, (int)o0)); r1 = max(0, min(255, (int)o1)); r2 = max(0, min(255, (int)o2));
    } else {
        r0 = ((int)o0) & 255; r1 = ((int)o1) & 255; r2 = ((int)o2) & 255;
    }
    p[0] = r0 * (1.f / 255.f); p[1] = r1 * (1.f / 255.f); p[2] = r2 * (1.f / 255.f);
}

// MLT brightness filter, yuv422 code path: luma and chroma scaled in 8-bit code space.
__global__ void brightness_yuv(float* img, int n, int m, float kr, float kb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* p = img + 4 * i;
    float kg = 1.f - kr - kb;
    float r = p[0], g = p[1], b = p[2];
    float Y = kr * r + kg * g + kb * b;
    int y8 = __float2int_rn(16.f + 219.f * Y);
    int u8 = __float2int_rn(128.f + 224.f * (b - Y) / (2.f * (1.f - kb)));
    int v8 = __float2int_rn(128.f + 224.f * (r - Y) / (2.f * (1.f - kr)));
    y8 = max(0, min(255, y8)); u8 = max(0, min(255, u8)); v8 = max(0, min(255, v8));
    long long nn = 128LL * (65536 - m);
    y8 = max(16, min(235, (int)(((long long)y8 * m) >> 16)));
    u8 = max(16, min(240, (int)(((long long)u8 * m + nn) >> 16)));
    v8 = max(16, min(240, (int)(((long long)v8 * m + nn) >> 16)));
    float yn = (y8 - 16.f) / 219.f, cb = (u8 - 128.f) / 224.f, cr = (v8 - 128.f) / 224.f;
    float ro = yn + 2.f * (1.f - kr) * cr;
    float bo = yn + 2.f * (1.f - kb) * cb;
    p[0] = ro; p[2] = bo; p[1] = (yn - kr * ro - kb * bo) / kg;
}

// Generic 3x4 colour matrix on 0..1 values (used by hue/eq style effects)
__global__ void color_matrix(float* img, int n, const float* m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* p = img + 4 * i;
    float r = p[0], g = p[1], b = p[2];
    p[0] = clamp01(m[0] * r + m[1] * g + m[2] * b + m[3]);
    p[1] = clamp01(m[4] * r + m[5] * g + m[6] * b + m[7]);
    p[2] = clamp01(m[8] * r + m[9] * g + m[10] * b + m[11]);
}

// ------------------------------------------------------------------ geometry / compositing
__device__ __forceinline__ float4 fetch(const float* src, int sw, int sh, int x, int y)
{
    if (x < 0 || y < 0 || x >= sw || y >= sh) return make_float4(0, 0, 0, 0);
    const float* p = src + 4 * (y * sw + x);
    return make_float4(p[0], p[1], p[2], p[3]);
}

// Bilinear sample returning PREMULTIPLIED colour (avoids dark fringes at alpha edges)
__device__ float4 sample_premul(const float* src, int sw, int sh, float sx, float sy, int nearest)
{
    if (nearest) {
        float4 c = fetch(src, sw, sh, (int)floorf(sx), (int)floorf(sy));
        return make_float4(c.x * c.w, c.y * c.w, c.z * c.w, c.w);
    }
    sx -= 0.5f; sy -= 0.5f;
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    float fx = sx - x0, fy = sy - y0;
    float4 a = fetch(src, sw, sh, x0, y0), b = fetch(src, sw, sh, x0 + 1, y0);
    float4 c = fetch(src, sw, sh, x0, y0 + 1), d = fetch(src, sw, sh, x0 + 1, y0 + 1);
    float wa = (1 - fx) * (1 - fy), wb = fx * (1 - fy), wc = (1 - fx) * fy, wd = fx * fy;
    float4 o;
    o.w = a.w * wa + b.w * wb + c.w * wc + d.w * wd;
    o.x = a.x * a.w * wa + b.x * b.w * wb + c.x * c.w * wc + d.x * d.w * wd;
    o.y = a.y * a.w * wa + b.y * b.w * wb + c.y * c.w * wc + d.y * d.w * wd;
    o.z = a.z * a.w * wa + b.z * b.w * wb + c.z * c.w * wc + d.z * d.w * wd;
    return o;
}

__device__ __forceinline__ float blend_fn(int mode, float cs, float cb)
{
    // W3C separable blend functions on straight colours
    switch (mode) {
    case 13: return cs * cb;                                                   // Multiply
    case 14: return cb + cs - cb * cs;                                         // Screen
    case 15: return (cb <= 0.5f) ? 2.f * cs * cb                               // Overlay
                                 : 1.f - 2.f * (1.f - cs) * (1.f - cb);
    case 16: return fminf(cs, cb);                                             // Darken
    case 17: return fmaxf(cs, cb);                                             // Lighten
    case 18: if (cb <= 0.f) return 0.f; if (cs >= 1.f) return 1.f;             // ColorDodge
             return fminf(1.f, cb / (1.f - cs));
    case 19: if (cb >= 1.f) return 1.f; if (cs <= 0.f) return 0.f;             // ColorBurn
             return 1.f - fminf(1.f, (1.f - cb) / cs);
    case 20: return (cs <= 0.5f) ? 2.f * cs * cb                               // HardLight
                                 : 1.f - 2.f * (1.f - cs) * (1.f - cb);
    case 21: {                                                                 // SoftLight
        if (cs <= 0.5f) return cb - (1.f - 2.f * cs) * cb * (1.f - cb);
        float D = (cb <= 0.25f) ? ((16.f * cb - 12.f) * cb + 4.f) * cb : sqrtf(cb);
        return cb + (2.f * cs - 1.f) * (D - cb); }
    case 22: return fabsf(cb - cs);                                            // Difference
    case 23: return cb + cs - 2.f * cb * cs;                                   // Exclusion
    }
    return cs;
}

__device__ __forceinline__ float blend_sep(int mode, float sc, float sa, float dc, float da)
{
    // premultiplied in/out
    if (mode == 12) return fminf(sc + dc, 1.f);                                // Plus
    float cs = sa > 0.f ? sc / sa : 0.f;
    float cb = da > 0.f ? dc / da : 0.f;
    return sc * (1.f - da) + dc * (1.f - sa) + sa * da * blend_fn(mode, clamp01(cs), clamp01(cb));
}

// Composite 'src' into 'dst' through an affine map (dst pixel -> src pixel coordinates).
// mode: Qt QPainter::CompositionMode; -1 = replace (used to place sources onto a fresh canvas)
__global__ void affine_composite(float* dst, int dw, int dh,
                                 const float* src, int sw, int sh,
                                 float m00, float m01, float m02, float m10, float m11, float m12,
                                 float opacity, int mode, int nearest)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    float px = x + 0.5f, py = y + 0.5f;
    float sx = m00 * px + m01 * py + m02;
    float sy = m10 * px + m11 * py + m12;
    float* d = dst + 4 * (y * dw + x);
    if (sx <= -1.f || sy <= -1.f || sx >= sw + 1.f || sy >= sh + 1.f) {
        if (mode == -1) { d[0] = d[1] = d[2] = d[3] = 0.f; }
        return;
    }
    float4 s = sample_premul(src, sw, sh, sx, sy, nearest);
    if (mode == -1) {
        float a = s.w;
        if (a > 1e-6f) { d[0] = s.x / a; d[1] = s.y / a; d[2] = s.z / a; }
        else { d[0] = d[1] = d[2] = 0.f; }
        d[3] = a;
        return;
    }
    float Sa = s.w * opacity;
    if (Sa <= 0.f && mode == 0) return;
    float Sr = s.x * opacity, Sg = s.y * opacity, Sb = s.z * opacity;
    float Da = d[3], Dr = d[0] * Da, Dg = d[1] * Da, Db = d[2] * Da;
    float Or, Og, Ob, Oa;
    switch (mode) {
    case 0:  Or = Sr + Dr * (1 - Sa); Og = Sg + Dg * (1 - Sa); Ob = Sb + Db * (1 - Sa); Oa = Sa + Da * (1 - Sa); break;
    case 1:  Or = Dr + Sr * (1 - Da); Og = Dg + Sg * (1 - Da); Ob = Db + Sb * (1 - Da); Oa = Da + Sa * (1 - Da); break;
    case 2:  Or = Og = Ob = Oa = 0.f; break;
    case 3:  Or = Sr; Og = Sg; Ob = Sb; Oa = Sa; break;
    case 4:  Or = Dr; Og = Dg; Ob = Db; Oa = Da; break;
    case 5:  Or = Sr * Da; Og = Sg * Da; Ob = Sb * Da; Oa = Sa * Da; break;
    case 6:  Or = Dr * Sa; Og = Dg * Sa; Ob = Db * Sa; Oa = Da * Sa; break;
    case 7:  Or = Sr * (1 - Da); Og = Sg * (1 - Da); Ob = Sb * (1 - Da); Oa = Sa * (1 - Da); break;
    case 8:  Or = Dr * (1 - Sa); Og = Dg * (1 - Sa); Ob = Db * (1 - Sa); Oa = Da * (1 - Sa); break;
    case 9:  Or = Sr * Da + Dr * (1 - Sa); Og = Sg * Da + Dg * (1 - Sa); Ob = Sb * Da + Db * (1 - Sa); Oa = Da; break;
    case 10: Or = Dr * Sa + Sr * (1 - Da); Og = Dg * Sa + Sg * (1 - Da); Ob = Db * Sa + Sb * (1 - Da); Oa = Sa; break;
    case 11: Or = Sr * (1 - Da) + Dr * (1 - Sa); Og = Sg * (1 - Da) + Dg * (1 - Sa);
             Ob = Sb * (1 - Da) + Db * (1 - Sa); Oa = Sa + Da - 2 * Sa * Da; break;
    default:
        Or = blend_sep(mode, Sr, Sa, Dr, Da); Og = blend_sep(mode, Sg, Sa, Dg, Da);
        Ob = blend_sep(mode, Sb, Sa, Db, Da);
        Oa = (mode == 12) ? fminf(Sa + Da, 1.f) : Sa + Da - Sa * Da;
    }
    Oa = clamp01(Oa);
    // Colours stay unclamped (MLT passes opaque YUV through untouched); 8-bit effects clamp themselves.
    if (Oa > 1e-6f) { d[0] = Or / Oa; d[1] = Og / Oa; d[2] = Ob / Oa; }
    else { d[0] = d[1] = d[2] = 0.f; }
    d[3] = Oa;
}

// Separable triangle-filter resize (swscale SWS_BILINEAR semantics; widens on downscale).
// horiz=1: src (sh, sw) -> dst (sh, dw); horiz=0: src (sh, sw) -> dst (dh, sw)
__global__ void resize_tri(const float* src, int sw, int sh, float* dst, int dw, int dh, int horiz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = horiz ? dw : sw, oh = horiz ? sh : dh;
    if (x >= ow || y >= oh) return;
    int sn = horiz ? sw : sh, dn = horiz ? dw : dh;
    float scale = (float)dn / (float)sn;              // < 1 when shrinking
    float support = scale < 1.f ? 1.f / scale : 1.f;
    float o = horiz ? x : y;
    float center = (o + 0.5f) / scale - 0.5f;
    int lo = (int)floorf(center - support) + 1, hi = (int)floorf(center + support);
    float r = 0, g = 0, b = 0, a = 0, wsum = 0;
    for (int t = lo; t <= hi; ++t) {
        float w = 1.f - fabsf(t - center) / support;
        if (w <= 0.f) continue;
        int tt = min(max(t, 0), sn - 1);
        const float* p = horiz ? src + 4 * (y * sw + tt) : src + 4 * (tt * sw + x);
        r += p[0] * p[3] * w; g += p[1] * p[3] * w; b += p[2] * p[3] * w; a += p[3] * w; wsum += w;
    }
    float* q = dst + 4 * (y * ow + x);
    if (a > 1e-6f) { q[0] = r / a; q[1] = g / a; q[2] = b / a; } else { q[0] = q[1] = q[2] = 0.f; }
    q[3] = wsum > 0.f ? a / wsum : 0.f;
}

// MLT luma transition, dissolve variant (dst = a, src = b)
__global__ void dissolve(float* dst, const float* src, int n, float weight)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* d = dst + 4 * i; const float* s = src + 4 * i;
    float mix_a = (1.f - weight) * d[3];
    float mix_b = weight * s[3];
    float mix2 = mix_a + mix_b;
    if (mix2 != 0.f) mix_b /= mix2;
    d[0] = s[0] * mix_b + d[0] * (1.f - mix_b);
    d[1] = s[1] * mix_b + d[1] * (1.f - mix_b);
    d[2] = s[2] * mix_b + d[2] * (1.f - mix_b);
    d[3] = clamp01(mix2);
}

// MLT luma transition with a wipe map (luma in 0..1, already resized to frame)
__global__ void luma_wipe(float* dst, const float* src, const float* luma, int n,
                          float pos, float softness)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* d = dst + 4 * i; const float* s = src + 4 * i;
    float L = luma[i];
    float e0 = L, e1 = L + softness, x = pos * (1.f + softness);
    float t = (e1 > e0) ? clamp01((x - e0) / (e1 - e0)) : (x >= e0 ? 1.f : 0.f);
    float value = t * t * (3.f - 2.f * t);
    float mix_a = (1.f - value) * d[3];
    float mix_b = value * s[3];
    float mix2 = mix_a + mix_b;
    if (mix2 != 0.f) mix_b /= mix2;
    d[0] = s[0] * mix_b + d[0] * (1.f - mix_b);
    d[1] = s[1] * mix_b + d[1] * (1.f - mix_b);
    d[2] = s[2] * mix_b + d[2] * (1.f - mix_b);
    d[3] = clamp01(mix2);
}

// Separable gaussian blur pass (horizontal when dx=1, vertical when dx=0)
__global__ void blur_pass(const float* src, float* dst, int w, int h, const float* k, int radius, int horiz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float r = 0, g = 0, b = 0, a = 0;
    for (int t = -radius; t <= radius; ++t) {
        int sx = horiz ? min(max(x + t, 0), w - 1) : x;
        int sy = horiz ? y : min(max(y + t, 0), h - 1);
        const float* p = src + 4 * (sy * w + sx);
        float kw = k[t + radius];
        r += p[0] * p[3] * kw; g += p[1] * p[3] * kw; b += p[2] * p[3] * kw; a += p[3] * kw;
    }
    float* o = dst + 4 * (y * w + x);
    if (a > 1e-6f) { o[0] = r / a; o[1] = g / a; o[2] = b / a; } else { o[0] = o[1] = o[2] = 0; }
    o[3] = a;
}

// ------------------------------------------------------------------ frei0r scale0tilt (gavl scaler)
// Separable resample driven by per-axis tap tables.  The source is the 8-bit RGBA frame MLT hands to
// frei0r, and the result is 8-bit too, so quantise on the way in and out.
__global__ void tap_pass_h(const float* src, int sw, int sh, float* dst, int dw,
                           const int* idx, const float* wgt, int ntaps)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= sh) return;
    float r = 0.f, g = 0.f, b = 0.f, a = 0.f;
    for (int k = 0; k < ntaps; ++k) {
        float w = wgt[x * ntaps + k];
        const float* p = src + 4 * (y * sw + idx[x * ntaps + k]);
        r += w * q8(p[0]); g += w * q8(p[1]); b += w * q8(p[2]); a += w * q8(p[3]);
    }
    float* o = dst + 4 * (y * dw + x);
    // integer pipeline: every pass truncates to an 8-bit code
    o[0] = floorf(fminf(r, 255.f)); o[1] = floorf(fminf(g, 255.f));
    o[2] = floorf(fminf(b, 255.f)); o[3] = floorf(fminf(a, 255.f));
}

__global__ void tap_pass_v(const float* tmp, int dw, float* dst, int out_w, int out_h,
                           int ox, int oy, int dh, const int* idx, const float* wgt, int ntaps)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    if (ox + x >= out_w || oy + y >= out_h) return;
    float r = 0.f, g = 0.f, b = 0.f, a = 0.f;
    for (int k = 0; k < ntaps; ++k) {
        float w = wgt[y * ntaps + k];
        const float* p = tmp + 4 * (idx[y * ntaps + k] * dw + x);
        r += w * p[0]; g += w * p[1]; b += w * p[2]; a += w * p[3];
    }
    float* o = dst + 4 * ((oy + y) * out_w + ox + x);
    const float k255 = 1.f / 255.f;
    o[0] = floorf(fminf(fmaxf(r, 0.f), 255.f)) * k255;
    o[1] = floorf(fminf(fmaxf(g, 0.f), 255.f)) * k255;
    o[2] = floorf(fminf(fmaxf(b, 0.f), 255.f)) * k255;
    o[3] = floorf(fminf(fmaxf(a, 0.f), 255.f)) * k255;
}

}  // extern "C"
"""

_module: cp.RawModule | None = None


CHROMA_MODE = int(os.environ.get("KNR_CHROMA", "2"))


def _mod() -> cp.RawModule:
    global _module
    if _module is None:
        _module = cp.RawModule(code=f"#define CHROMA_MODE {CHROMA_MODE}\n" + _SRC,
                               options=("--use_fast_math",))
    return _module


def _k(name: str):
    return _mod().get_function(name)


def _grid1(n, block=256):
    return ((n + block - 1) // block,), (block,)


def _grid2(w, h, bx=32, by=8):
    return ((w + bx - 1) // bx, (h + by - 1) // by), (bx, by)


def warmup():
    _mod()


# ------------------------------------------------------------------ helpers

def new_canvas(w: int, h: int) -> cp.ndarray:
    return cp.zeros((h, w, 4), dtype=cp.float32)


def solid(w: int, h: int, rgba: tuple[float, float, float, float]) -> cp.ndarray:
    img = cp.empty((h, w, 4), dtype=cp.float32)
    img[...] = cp.asarray(rgba, dtype=cp.float32)
    return img


def coeffs(colorspace: int) -> tuple[float, float]:
    if colorspace in (601, 470, 170, 5, 6):
        return 0.299, 0.114
    if colorspace in (2020, 9, 10):
        return 0.2627, 0.0593
    if colorspace == 240:
        return 0.212, 0.087
    return 0.2126, 0.0722


def yuv420_to_rgba(raw: cp.ndarray, w: int, h: int, is16: bool, colorspace: int,
                   full_range: bool) -> cp.ndarray:
    """``raw`` is the flat NV12/P010 buffer as delivered by FFmpeg."""
    kr, kb = coeffs(colorspace)
    out = cp.empty((h, w, 4), dtype=cp.float32)
    grid, block = _grid2(w, h)
    if is16:
        buf = raw.view(cp.uint16)
        y = buf[: w * h]
        uv = buf[w * h:]
        dummy = cp.empty(1, dtype=cp.uint8)
        _k("yuv420_to_rgba")(grid, block, (dummy, dummy, y, uv, np.int32(w), np.int32(h), np.int32(w),
                                           np.int32(w), np.float32(kr), np.float32(kb),
                                           np.int32(full_range), np.int32(1), out))
    else:
        y = raw[: w * h]
        uv = raw[w * h:]
        dummy = cp.empty(1, dtype=cp.uint16)
        _k("yuv420_to_rgba")(grid, block, (y, uv, dummy, dummy, np.int32(w), np.int32(h), np.int32(w),
                                           np.int32(w), np.float32(kr), np.float32(kb),
                                           np.int32(full_range), np.int32(0), out))
    return out


def rgba8_to_rgba(raw: cp.ndarray, w: int, h: int) -> cp.ndarray:
    out = cp.empty((h, w, 4), dtype=cp.float32)
    grid, block = _grid1(w * h * 4)
    _k("rgba8_to_rgba")(grid, block, (raw, np.int32(w * h), out))
    return out


def rgba_to_nv12(img: cp.ndarray, colorspace: int, out: cp.ndarray | None = None) -> cp.ndarray:
    h, w = img.shape[:2]
    kr, kb = coeffs(colorspace)
    if out is None:
        out = cp.empty(w * h * 3 // 2, dtype=cp.uint8)
    grid, block = _grid2(w // 2, h // 2)
    _k("rgba_to_nv12")(grid, block, (img, np.int32(w), np.int32(h), np.float32(kr), np.float32(kb),
                                     out[: w * h], out[w * h:]))
    return out


def rgba_to_yuv444(img: cp.ndarray, colorspace: int) -> cp.ndarray:
    h, w = img.shape[:2]
    kr, kb = coeffs(colorspace)
    out = cp.empty(w * h * 3, dtype=cp.uint8)
    grid, block = _grid2(w, h)
    _k("rgba_to_yuv444")(grid, block, (img, np.int32(w), np.int32(h), np.float32(kr), np.float32(kb), out))
    return out


def apply_lut3(img: cp.ndarray, lut: np.ndarray):
    """lut: uint8 array of shape (3, 256)."""
    n = img.shape[0] * img.shape[1]
    grid, block = _grid1(n)
    _k("apply_lut3")(grid, block, (img, np.int32(n), cp.asarray(np.ascontiguousarray(lut, dtype=np.uint8))))


def apply_alpha_lut(img: cp.ndarray, lut: np.ndarray):
    n = img.shape[0] * img.shape[1]
    grid, block = _grid1(n)
    _k("apply_alpha_lut")(grid, block, (img, np.int32(n), cp.asarray(np.ascontiguousarray(lut, dtype=np.uint8))))


def saturat0r(img: cp.ndarray, saturation: float):
    n = img.shape[0] * img.shape[1]
    one_minus = 1.0 - saturation
    w0 = int(7471.0 * one_minus)
    w1 = int(38470.0 * one_minus)
    w2 = int(19595.0 * one_minus)
    clampit = 0 if 0 <= saturation <= 1 else 1
    grid, block = _grid1(n)
    _k("saturat0r")(grid, block, (img, np.int32(n), np.int32(w0), np.int32(w1), np.int32(w2),
                                  np.float32(saturation), np.int32(clampit)))


def brightness_yuv(img: cp.ndarray, level: float, colorspace: int = 709):
    n = img.shape[0] * img.shape[1]
    kr, kb = coeffs(colorspace)
    grid, block = _grid1(n)
    _k("brightness_yuv")(grid, block, (img, np.int32(n), np.int32(int(level * 65536)),
                                       np.float32(kr), np.float32(kb)))


def color_matrix(img: cp.ndarray, m: np.ndarray):
    n = img.shape[0] * img.shape[1]
    grid, block = _grid1(n)
    _k("color_matrix")(grid, block, (img, np.int32(n), cp.asarray(np.asarray(m, dtype=np.float32).ravel())))


MIPMAP_STEP = 0.85     # MLT_QT_MIPMAP_STEP


def resize(src: cp.ndarray, dw: int, dh: int) -> cp.ndarray:
    """Triangle-filtered resize (matches swscale bilinear incl. its widening on downscale)."""
    sh, sw = src.shape[:2]
    tmp = src
    if dw != sw:
        tmp = cp.empty((sh, dw, 4), dtype=cp.float32)
        grid, block = _grid2(dw, sh)
        _k("resize_tri")(grid, block, (src, np.int32(sw), np.int32(sh), tmp, np.int32(dw), np.int32(dh), np.int32(1)))
    if dh != sh:
        out = cp.empty((dh, dw, 4), dtype=cp.float32)
        grid, block = _grid2(dw, dh)
        _k("resize_tri")(grid, block, (tmp, np.int32(dw), np.int32(sh), out, np.int32(dw), np.int32(dh), np.int32(0)))
        tmp = out
    return tmp


def _mip(src: cp.ndarray, scale: float) -> tuple[cp.ndarray, float, float]:
    """Pre-shrink like MLT's qtblend: largest 0.85^n step that is still >= the target scale."""
    m = 1.0
    while m * MIPMAP_STEP >= scale and m > 0.001:
        m *= MIPMAP_STEP
    if m >= 1.0:
        return src, 1.0, 1.0
    sh, sw = src.shape[:2]
    nw, nh = max(1, int(round(sw * m))), max(1, int(round(sh * m)))
    return resize(src, nw, nh), sw / nw, sh / nh


def affine_composite(dst: cp.ndarray, src: cp.ndarray, forward: np.ndarray, opacity: float = 1.0,
                     mode: int = 0, nearest: bool = False):
    """Draw ``src`` onto ``dst`` using the 3x3 ``forward`` matrix (src px -> dst px)."""
    dh, dw = dst.shape[:2]
    # estimate minification for mip-mapping
    sx = math.hypot(forward[0, 0], forward[1, 0])
    sy = math.hypot(forward[0, 1], forward[1, 1])
    scale = min(sx, sy)
    if not nearest and scale < 1.0:
        src, fx, fy = _mip(src, scale)
        forward = forward @ np.diag([fx, fy, 1.0])
    sh, sw = src.shape[:2]
    inv = np.linalg.inv(forward)
    grid, block = _grid2(dw, dh)
    _k("affine_composite")(grid, block, (dst, np.int32(dw), np.int32(dh), src, np.int32(sw), np.int32(sh),
                                         np.float32(inv[0, 0]), np.float32(inv[0, 1]), np.float32(inv[0, 2]),
                                         np.float32(inv[1, 0]), np.float32(inv[1, 1]), np.float32(inv[1, 2]),
                                         np.float32(opacity), np.int32(mode), np.int32(1 if nearest else 0)))


def dissolve(a: cp.ndarray, b: cp.ndarray, weight: float):
    n = a.shape[0] * a.shape[1]
    grid, block = _grid1(n)
    _k("dissolve")(grid, block, (a, b, np.int32(n), np.float32(weight)))


def luma_wipe(a: cp.ndarray, b: cp.ndarray, luma: cp.ndarray, pos: float, softness: float):
    n = a.shape[0] * a.shape[1]
    grid, block = _grid1(n)
    _k("luma_wipe")(grid, block, (a, b, luma, np.int32(n), np.float32(pos), np.float32(softness)))


def gaussian_blur(img: cp.ndarray, sigma: float) -> cp.ndarray:
    if sigma <= 0.05:
        return img
    radius = max(1, int(math.ceil(sigma * 3)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(xs ** 2) / (2 * sigma * sigma))
    k = cp.asarray((k / k.sum()).astype(np.float32))
    h, w = img.shape[:2]
    tmp = cp.empty_like(img)
    out = cp.empty_like(img)
    grid, block = _grid2(w, h)
    _k("blur_pass")(grid, block, (img, tmp, np.int32(w), np.int32(h), k, np.int32(radius), np.int32(1)))
    _k("blur_pass")(grid, block, (tmp, out, np.int32(w), np.int32(h), k, np.int32(radius), np.int32(0)))
    return out


def _axis_taps(n_out: int, n_in: int, src0: float, srclen: float) -> tuple[np.ndarray, np.ndarray]:
    """Tap table of gavl's scaler (as measured against MLT's frei0r scale0tilt).

    Triangle filter of radius R = ceil(scale ratio) (1 when enlarging) centred on the pixel-centre
    source position; taps cover [round(c) - R, round(c) + R - 1]; positions outside the picture
    clamp to the edge pixel; weights are normalised and truncated to 16-bit fixed point.  Each pass
    truncates its 8-bit result (the scaler is an integer pipeline).
    """
    ratio = srclen / n_out
    R = max(1, int(math.ceil(ratio - 1e-9)))
    c = src0 + (np.arange(n_out) + 0.5) * ratio - 0.5
    rc = np.floor(c + 0.5).astype(np.int64)
    idx = rc[:, None] - R + np.arange(2 * R)[None, :]
    wgt = np.maximum(0.0, 1.0 - np.abs(idx - c[:, None]) / R)
    wgt /= wgt.sum(axis=1, keepdims=True)
    wgt = np.floor(wgt * 65536.0) / 65536.0          # 16-bit fixed-point weights (truncated)
    return np.clip(idx, 0, n_in - 1).astype(np.int32), wgt.astype(np.float32)


def resample_rect(img: cp.ndarray, src_rect: tuple[float, float, float, float],
                  dst_rect: tuple[int, int, int, int]) -> cp.ndarray:
    """Scale ``src_rect`` (float pixels) of ``img`` into ``dst_rect`` (ints) of a transparent frame."""
    h, w = img.shape[:2]
    sx, sy, sw, sh = src_rect
    dx, dy, dw, dh = dst_rect
    out = cp.zeros_like(img)
    if dw <= 0 or dh <= 0 or dx >= w or dy >= h:
        return out
    ix, wx = _axis_taps(dw, w, sx, sw)
    iy, wy = _axis_taps(dh, h, sy, sh)
    ix, wx, iy, wy = (cp.asarray(a) for a in (ix, wx, iy, wy))
    tmp = cp.empty((h, dw, 4), dtype=cp.float32)
    grid, block = _grid2(dw, h)
    _k("tap_pass_h")(grid, block, (img, np.int32(w), np.int32(h), tmp, np.int32(dw), ix, wx,
                                   np.int32(ix.shape[1])))
    grid, block = _grid2(dw, dh)
    _k("tap_pass_v")(grid, block, (tmp, np.int32(dw), out, np.int32(w), np.int32(h), np.int32(dx),
                                   np.int32(dy), np.int32(dh), iy, wy, np.int32(iy.shape[1])))
    return out
