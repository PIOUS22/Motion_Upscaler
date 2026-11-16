/* ============================================================================
 * test_engine.c -- self-contained correctness tests for the upscaler engine.
 *
 * Buffers handed to the engine are malloc'd with EXACTLY (h-1)*stride + w*ch
 * bytes and padding bytes are pre-filled with a sentinel, so under Valgrind /
 * ASan any read past the last pixel or any write into row padding is caught.
 * No external files, no stb: run it anywhere.
 * ==========================================================================*/
#include "upscaler_engine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_fail;
#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        ++g_checks;                                                           \
        if (!(cond)) {                                                        \
            ++g_fail;                                                         \
            printf("    FAIL %s:%d: ", __FILE__, __LINE__);                   \
            printf(__VA_ARGS__);                                              \
            printf("\n");                                                     \
        }                                                                     \
    } while (0)

static uint32_t g_rng = 0xC0FFEEu;
static uint32_t rnd(void) { g_rng = g_rng * 1664525u + 1013904223u; return g_rng >> 8; }

/* ---- exact-size strided buffers -------------------------------------------- */
#define PAD_SENTINEL 0xAA

typedef struct Exact { ImageBuffer v; uint8_t *mem; size_t size; } Exact;

static void exact_alloc(Exact *e, int w, int h, int ch, int pad)
{
    const int stride = w * ch + pad;
    e->size = (size_t)(h - 1) * (size_t)stride + (size_t)w * (size_t)ch;
    e->mem = (uint8_t *)malloc(e->size);
    memset(e->mem, PAD_SENTINEL, e->size);
    e->v.data = e->mem; e->v.width = w; e->v.height = h; e->v.channels = ch; e->v.stride = stride;
}
static void exact_free(Exact *e) { free(e->mem); memset(e, 0, sizeof *e); }

static uint8_t *px(const Exact *e, int x, int y) { return e->mem + (size_t)y * (size_t)e->v.stride + (size_t)x * (size_t)e->v.channels; }

static void fill_random(Exact *e)
{
    for (int y = 0; y < e->v.height; ++y)
        for (int i = 0; i < e->v.width * e->v.channels; ++i)
            e->mem[(size_t)y * (size_t)e->v.stride + (size_t)i] = (uint8_t)(rnd() & 0xFF);
}
static void fill_const(Exact *e, uint8_t v)
{
    for (int y = 0; y < e->v.height; ++y)
        memset(e->mem + (size_t)y * (size_t)e->v.stride, v, (size_t)e->v.width * (size_t)e->v.channels);
}
static int padding_intact(const Exact *e)
{
    for (int y = 0; y < e->v.height; ++y) {
        const size_t row_end = (size_t)y * (size_t)e->v.stride + (size_t)e->v.width * (size_t)e->v.channels;
        const size_t next    = (y + 1 < e->v.height) ? (size_t)(y + 1) * (size_t)e->v.stride : row_end;
        for (size_t i = row_end; i < next; ++i)
            if (e->mem[i] != PAD_SENTINEL) return 0;
    }
    return 1;
}

/* ---- slow, obvious, double-precision reference ------------------------------ */
static double ref_catmull(double x)
{
    x = fabs(x);
    if (x < 1.0) return 1.5 * x * x * x - 2.5 * x * x + 1.0;
    if (x < 2.0) return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
    return 0.0;
}
static double ref_sinc(double x) { if (fabs(x) < 1e-12) return 1.0; const double p = 3.14159265358979323846 * x; return sin(p) / p; }
static double ref_lanczos(double x) { x = fabs(x); return x < 3.0 ? ref_sinc(x) * ref_sinc(x / 3.0) : 0.0; }
static uint8_t ref_round8(double v) { v = floor(v + 0.5); return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }
static int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* Direct evaluation of the kernel at (x+0.5)/s-0.5 for every output pixel, edge
 * replication via index clamping, 8-bit rounding after the horizontal pass (the
 * same documented contract as the engine). Shares NO code with the engine.     */
static void ref_upscale(const Exact *src, int scale, int kernel, uint8_t *out /* tight */)
{
    const int w = src->v.width, h = src->v.height, ch = src->v.channels;
    const int dw = w * scale, dh = h * scale;
    const int taps = kernel == UPSCALE_KERNEL_LANCZOS3 ? 6 : 4, left = taps / 2 - 1;
    uint8_t *mid = (uint8_t *)malloc((size_t)dw * (size_t)h * (size_t)ch);

    for (int y = 0; y < h; ++y)
        for (int ox = 0; ox < dw; ++ox) {
            const double sx = ((double)ox + 0.5) / scale - 0.5;
            const int b = (int)floor(sx);
            double wt[6], sum = 0.0;
            for (int t = 0; t < taps; ++t) {
                const double d = sx - (double)(b - left + t);
                wt[t] = kernel == UPSCALE_KERNEL_LANCZOS3 ? ref_lanczos(d) : ref_catmull(d);
                sum += wt[t];
            }
            for (int c = 0; c < ch; ++c) {
                double acc = 0.0;
                for (int t = 0; t < taps; ++t)
                    acc += wt[t] / sum * px(src, clampi(b - left + t, 0, w - 1), y)[c];
                mid[((size_t)y * (size_t)dw + (size_t)ox) * (size_t)ch + (size_t)c] = ref_round8(acc);
            }
        }
    for (int oy = 0; oy < dh; ++oy) {
        const double sy = ((double)oy + 0.5) / scale - 0.5;
        const int b = (int)floor(sy);
        double wt[6], sum = 0.0;
        for (int t = 0; t < taps; ++t) {
            const double d = sy - (double)(b - left + t);
            wt[t] = kernel == UPSCALE_KERNEL_LANCZOS3 ? ref_lanczos(d) : ref_catmull(d);
            sum += wt[t];
        }
        for (int ox = 0; ox < dw; ++ox)
            for (int c = 0; c < ch; ++c) {
                double acc = 0.0;
                for (int t = 0; t < taps; ++t)
                    acc += wt[t] / sum * mid[((size_t)clampi(b - left + t, 0, h - 1) * (size_t)dw + (size_t)ox) * (size_t)ch + (size_t)c];
                out[((size_t)oy * (size_t)dw + (size_t)ox) * (size_t)ch + (size_t)c] = ref_round8(acc);
            }
    }
    free(mid);
}

/* ============================================================================
 *  T1: engine == double-precision reference (indexing, phases, edges, strides)
 * ==========================================================================*/
static void test_against_reference(void)
{
    printf("[T1] resampler vs. double-precision reference\n");
    static const int sizes[][2] = { {1,1}, {1,7}, {7,1}, {2,2}, {5,5}, {17,13}, {40,31} };
    int worst = 0;
    for (size_t si = 0; si < sizeof sizes / sizeof sizes[0]; ++si)
    for (int ch = 1; ch <= 4; ++ch)
    for (int scale = 2; scale <= 4; scale += 2)
    for (int kernel = 0; kernel <= 1; ++kernel)
    for (int pad = 0; pad <= 11; pad += 11) {
        const int w = sizes[si][0], h = sizes[si][1], dw = w * scale, dh = h * scale;
        Exact s, d;
        exact_alloc(&s, w, h, ch, pad);
        exact_alloc(&d, dw, dh, ch, pad ? 5 : 0);
        fill_random(&s);
        uint8_t *ref = (uint8_t *)malloc((size_t)dw * (size_t)dh * (size_t)ch);
        ref_upscale(&s, scale, kernel, ref);

        const int32_t rc = upscale_resample_rows(&s.v, scale, kernel, &d.v, 0, dh);
        CHECK(rc == UPSCALE_OK, "rc=%d %dx%d ch%d s%d k%d", rc, w, h, ch, scale, kernel);

        int maxd = 0;
        for (int y = 0; y < dh; ++y)
            for (int i = 0; i < dw * ch; ++i) {
                const int diff = abs((int)px(&d, 0, y)[i] - (int)ref[(size_t)y * (size_t)dw * (size_t)ch + (size_t)i]);
                if (diff > maxd) maxd = diff;
            }
        if (maxd > worst) worst = maxd;
        /* Error budget: the engine rounds the intermediate image with Q14 weights, the reference
         * with doubles, so an intermediate pixel may differ by 1 level at a .5 boundary. The
         * vertical pass amplifies that by sum|w| (1.13 Catmull-Rom, 1.26 Lanczos-3) and adds one
         * more rounding => <= 2 levels for Lanczos, <= 1 for Catmull-Rom on worst-case noise. */
        const int tol = kernel == UPSCALE_KERNEL_LANCZOS3 ? 2 : 1;
        CHECK(maxd <= tol, "max diff %d (>%d) %dx%d ch%d s%d k%d pad%d", maxd, tol, w, h, ch, scale, kernel, pad);
        CHECK(padding_intact(&d), "engine wrote into row padding %dx%d ch%d", w, h, ch);
        CHECK(padding_intact(&s), "engine modified source padding");
        free(ref); exact_free(&s); exact_free(&d);
    }
    printf("     worst |engine - reference| over all cases: %d level(s)\n", worst);
}

/* ============================================================================
 *  T2: flat regions are reproduced bit-exactly (weights sum to exactly 1)
 * ==========================================================================*/
static void test_dc_preservation(void)
{
    printf("[T2] constant images stay constant (incl. sharpen)\n");
    static const uint8_t vals[] = { 0, 1, 127, 254, 255 };
    for (size_t vi = 0; vi < sizeof vals; ++vi)
    for (int ch = 1; ch <= 4; ++ch)
    for (int scale = 2; scale <= 4; scale += 2)
    for (int kernel = 0; kernel <= 1; ++kernel) {
        Exact s, d;
        exact_alloc(&s, 9, 7, ch, 3);
        exact_alloc(&d, 9 * scale, 7 * scale, ch, 0);
        fill_const(&s, vals[vi]);
        UpscaleOptions o; upscaler_options_default(&o);
        o.scale = scale; o.kernel = kernel; o.sharpen_amount = 1.5f; o.sharpen_radius = 1.2f;
        const int32_t rc = upscale_image_into(&s.v, &o, &d.v);
        int bad = (rc != UPSCALE_OK);
        for (int y = 0; y < d.v.height && !bad; ++y)
            for (int i = 0; i < d.v.width * ch; ++i)
                if (px(&d, 0, y)[i] != vals[vi]) { bad = 1; break; }
        CHECK(!bad, "value %u ch%d scale%d kernel%d not preserved", vals[vi], ch, scale, kernel);
        exact_free(&s); exact_free(&d);
    }
}

/* ============================================================================
 *  T3: row bands are bit-identical to one full pass (thread-safe splitting)
 * ==========================================================================*/
static void test_banding(void)
{
    printf("[T3] banded resampling == full resampling\n");
    for (int kernel = 0; kernel <= 1; ++kernel)
    for (int scale = 2; scale <= 4; scale += 2) {
        const int w = 33, h = 29, dw = w * scale, dh = h * scale;
        Exact s, full, band;
        exact_alloc(&s, w, h, 4, 0);
        exact_alloc(&full, dw, dh, 4, 0);
        exact_alloc(&band, dw, dh, 4, 0);
        fill_random(&s);
        CHECK(upscale_resample_rows(&s.v, scale, kernel, &full.v, 0, dh) == UPSCALE_OK, "full");
        const int cuts[] = { 0, 1, 17, 18, dh / 2 + 1, dh - 1, dh };
        for (size_t i = 0; i + 1 < sizeof cuts / sizeof cuts[0]; ++i)
            CHECK(upscale_resample_rows(&s.v, scale, kernel, &band.v, cuts[i], cuts[i + 1]) == UPSCALE_OK, "band %zu", i);
        CHECK(memcmp(full.mem, band.mem, full.size) == 0, "banded result differs (scale %d kernel %d)", scale, kernel);
        exact_free(&s); exact_free(&full); exact_free(&band);
    }
}

/* ============================================================================
 *  T4: unsharp mask -- reference match, behaviour, alpha, threshold, edges
 * ==========================================================================*/
static void ref_unsharp(Exact *img, float amount, float radius, int thr)
{
    const int w = img->v.width, h = img->v.height, ch = img->v.channels;
    const int ncolor = (ch == 2 || ch == 4) ? ch - 1 : ch;
    int R = (int)ceil(3.0 * radius); R = clampi(R, 1, 15);
    double g[31], sum = 0.0;
    for (int i = 0; i < 2 * R + 1; ++i) { g[i] = exp(-0.5 * (i - R) * (i - R) / ((double)radius * radius)); sum += g[i]; }
    for (int i = 0; i < 2 * R + 1; ++i) g[i] /= sum;

    uint8_t *hb = (uint8_t *)malloc((size_t)w * (size_t)h * (size_t)ch);
    uint8_t *vb = (uint8_t *)malloc((size_t)w * (size_t)h * (size_t)ch);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < ch; ++c) {
                double a = 0;
                for (int k = 0; k < 2 * R + 1; ++k) a += g[k] * px(img, clampi(x - R + k, 0, w - 1), y)[c];
                hb[((size_t)y * (size_t)w + (size_t)x) * (size_t)ch + (size_t)c] = ref_round8(a);
            }
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < ch; ++c) {
                double a = 0;
                for (int k = 0; k < 2 * R + 1; ++k)
                    a += g[k] * hb[((size_t)clampi(y - R + k, 0, h - 1) * (size_t)w + (size_t)x) * (size_t)ch + (size_t)c];
                vb[((size_t)y * (size_t)w + (size_t)x) * (size_t)ch + (size_t)c] = ref_round8(a);
            }
    const int amt_q8 = (int)(amount * 256.0f + 0.5f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < ncolor; ++c) {
                const int o = px(img, x, y)[c];
                const int d = o - vb[((size_t)y * (size_t)w + (size_t)x) * (size_t)ch + (size_t)c];
                if (d > thr || d < -thr) {
                    int v = o + ((d * amt_q8 + 128) >> 8);
                    px(img, x, y)[c] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
            }
    free(hb); free(vb);
}

static void test_unsharp(void)
{
    printf("[T4] unsharp mask\n");
    int worst = 0;
    static const int dims[][2] = { {20, 17}, {3, 2}, {1, 9}, {64, 5} };
    static const float radii[] = { 0.5f, 1.0f, 1.8f, 5.0f };
    for (size_t di = 0; di < 4; ++di)
    for (size_t ri = 0; ri < 4; ++ri)
    for (int ch = 1; ch <= 4; ++ch) {
        Exact a, b;
        exact_alloc(&a, dims[di][0], dims[di][1], ch, 7);
        exact_alloc(&b, dims[di][0], dims[di][1], ch, 7);
        fill_random(&a);
        for (int y = 0; y < a.v.height; ++y)
            memcpy(b.mem + (size_t)y * (size_t)b.v.stride, a.mem + (size_t)y * (size_t)a.v.stride, (size_t)a.v.width * (size_t)ch);
        /* threshold 0 here: a non-zero threshold is a hard step in the transfer function, so a
         * 1-level blur rounding difference can legitimately flip it (tested separately below). */
        CHECK(upscale_unsharp_mask(&a.v, 1.0f, radii[ri], 0) == UPSCALE_OK, "unsharp rc");
        ref_unsharp(&b, 1.0f, radii[ri], 0);
        int maxd = 0;
        for (int y = 0; y < a.v.height; ++y)
            for (int i = 0; i < a.v.width * ch; ++i) {
                const int diff = abs((int)px(&a, 0, y)[i] - (int)px(&b, 0, y)[i]);
                if (diff > maxd) maxd = diff;
            }
        if (maxd > worst) worst = maxd;
        /* Blur differs by <= 2 (two rounded 8-bit stages), times amount 1.0, plus final rounding. */
        CHECK(maxd <= 3, "unsharp vs ref diff %d (dims %dx%d ch%d r%.1f)", maxd, dims[di][0], dims[di][1], ch, (double)radii[ri]);
        CHECK(padding_intact(&a), "unsharp wrote into padding");
        exact_free(&a); exact_free(&b);
    }
    printf("     worst |engine - reference|: %d level(s)\n", worst);

    { /* step edge: undershoot on the dark side, overshoot on the bright side */
        Exact e; exact_alloc(&e, 16, 8, 1, 0);
        for (int y = 0; y < 8; ++y) for (int x = 0; x < 16; ++x) *px(&e, x, y) = x < 8 ? 50 : 200;
        CHECK(upscale_unsharp_mask(&e.v, 1.5f, 1.0f, 0) == UPSCALE_OK, "edge rc");
        CHECK(*px(&e, 7, 4) < 50 && *px(&e, 8, 4) > 200, "no edge overshoot: %u / %u", *px(&e, 7, 4), *px(&e, 8, 4));
        CHECK(*px(&e, 0, 4) == 50 && *px(&e, 15, 4) == 200, "flat areas changed");
        exact_free(&e);
    }
    { /* threshold semantics: a small step (10 levels) is left alone at thr=20, sharpened at thr=2 */
        Exact a, b; exact_alloc(&a, 16, 8, 1, 0); exact_alloc(&b, 16, 8, 1, 0);
        for (int y = 0; y < 8; ++y) for (int x = 0; x < 16; ++x) *px(&a, x, y) = *px(&b, x, y) = x < 8 ? 100 : 110;
        CHECK(upscale_unsharp_mask(&a.v, 2.0f, 1.0f, 20) == UPSCALE_OK, "thr rc");
        CHECK(upscale_unsharp_mask(&b.v, 2.0f, 1.0f, 2) == UPSCALE_OK, "thr rc");
        CHECK(*px(&a, 7, 4) == 100 && *px(&a, 8, 4) == 110, "threshold 20 should leave a 10-level edge alone");
        CHECK(*px(&b, 7, 4) < 100 && *px(&b, 8, 4) > 110, "threshold 2 should sharpen a 10-level edge");
        exact_free(&a); exact_free(&b);
    }
    { /* alpha channel is never touched */
        Exact e, o; exact_alloc(&e, 12, 12, 4, 0); exact_alloc(&o, 12, 12, 4, 0);
        fill_random(&e); memcpy(o.mem, e.mem, e.size);
        CHECK(upscale_unsharp_mask(&e.v, 2.0f, 1.0f, 0) == UPSCALE_OK, "alpha rc");
        int alpha_same = 1, rgb_changed = 0;
        for (int i = 0; i < 12 * 12; ++i) {
            if (e.mem[i * 4 + 3] != o.mem[i * 4 + 3]) alpha_same = 0;
            if (e.mem[i * 4] != o.mem[i * 4]) rgb_changed = 1;
        }
        CHECK(alpha_same, "alpha modified"); CHECK(rgb_changed, "rgb not modified");
        exact_free(&e); exact_free(&o);
    }
    { /* amount 0 and threshold 255 are no-ops */
        Exact e, o; exact_alloc(&e, 10, 10, 3, 0); exact_alloc(&o, 10, 10, 3, 0);
        fill_random(&e); memcpy(o.mem, e.mem, e.size);
        CHECK(upscale_unsharp_mask(&e.v, 0.0f, 1.0f, 0) == UPSCALE_OK && memcmp(e.mem, o.mem, e.size) == 0, "amount 0 changed data");
        CHECK(upscale_unsharp_mask(&e.v, 3.0f, 1.0f, 255) == UPSCALE_OK && memcmp(e.mem, o.mem, e.size) == 0, "threshold 255 changed data");
        exact_free(&e); exact_free(&o);
    }
}

/* ============================================================================
 *  T5: bad input never crashes and yields the documented error code
 * ==========================================================================*/
static void test_invalid_args(void)
{
    printf("[T5] error handling\n");
    uint8_t spx[4 * 4 * 4] = {0}, dpx[8 * 8 * 4] = {0};
    const ImageBuffer s0 = { spx, 4, 4, 4, 16 }, d0 = { dpx, 8, 8, 4, 32 };
    UpscaleOptions o; upscaler_options_default(&o);
    ImageBuffer s, d;

    CHECK(upscale_image_into(NULL, &o, &d0) == UPSCALE_ERR_INVALID_ARGS, "null src");
    CHECK(upscale_image_into(&s0, NULL, &d0) == UPSCALE_ERR_INVALID_ARGS, "null opts");
    CHECK(upscale_image_into(&s0, &o, NULL) == UPSCALE_ERR_INVALID_ARGS, "null dst");
    CHECK(upscale_image_into(&s0, &o, &d0) == UPSCALE_OK, "baseline must succeed");

    o.scale = 3;  CHECK(upscale_image_into(&s0, &o, &d0) == UPSCALE_ERR_UNSUPPORTED, "scale 3");
    o.scale = 2;  o.kernel = 9;  CHECK(upscale_image_into(&s0, &o, &d0) == UPSCALE_ERR_UNSUPPORTED, "kernel 9");
    upscaler_options_default(&o);
    o.sharpen_amount = NAN;      CHECK(upscale_image_into(&s0, &o, &d0) == UPSCALE_ERR_INVALID_ARGS, "NaN amount");
    upscaler_options_default(&o);
    o.sharpen_amount = -1.0f;    CHECK(upscale_image_into(&s0, &o, &d0) == UPSCALE_ERR_INVALID_ARGS, "neg amount");
    upscaler_options_default(&o);
    o.sharpen_radius = 0.0f;     CHECK(upscale_image_into(&s0, &o, &d0) == UPSCALE_ERR_INVALID_ARGS, "radius 0");
    upscaler_options_default(&o);
    o.sharpen_threshold = 256;   CHECK(upscale_image_into(&s0, &o, &d0) == UPSCALE_ERR_INVALID_ARGS, "threshold 256");
    upscaler_options_default(&o);
    o.flags = 2;                 CHECK(upscale_image_into(&s0, &o, &d0) == UPSCALE_ERR_INVALID_ARGS, "unknown flag");
    upscaler_options_default(&o);

    s = s0; s.channels = 5;      CHECK(upscale_image_into(&s, &o, &d0) == UPSCALE_ERR_UNSUPPORTED, "5 channels");
    s = s0; s.channels = 0;      CHECK(upscale_image_into(&s, &o, &d0) == UPSCALE_ERR_UNSUPPORTED, "0 channels");
    s = s0; s.width = 0;         CHECK(upscale_image_into(&s, &o, &d0) == UPSCALE_ERR_INVALID_ARGS, "width 0");
    s = s0; s.height = -3;       CHECK(upscale_image_into(&s, &o, &d0) == UPSCALE_ERR_INVALID_ARGS, "height < 0");
    s = s0; s.stride = 15;       CHECK(upscale_image_into(&s, &o, &d0) == UPSCALE_ERR_INVALID_ARGS, "stride < row");
    s = s0; s.data = NULL;       CHECK(upscale_image_into(&s, &o, &d0) == UPSCALE_ERR_INVALID_ARGS, "null data");
    s = s0; s.width = 40000;     CHECK(upscale_image_into(&s, &o, &d0) == UPSCALE_ERR_TOO_LARGE, "width too large");
    d = d0; d.width = 7;         CHECK(upscale_image_into(&s0, &o, &d) == UPSCALE_ERR_INVALID_ARGS, "dst width mismatch");
    d = d0; d.channels = 3;      CHECK(upscale_image_into(&s0, &o, &d) == UPSCALE_ERR_INVALID_ARGS, "dst channel mismatch");
    d = d0; d.data = spx;        CHECK(upscale_image_into(&s0, &o, &d) == UPSCALE_ERR_OVERLAP, "dst == src");
    d = d0; d.data = spx + 8;    CHECK(upscale_image_into(&s0, &o, &d) == UPSCALE_ERR_OVERLAP, "partial overlap");

    CHECK(upscale_resample_rows(&s0, 2, 0, &d0, -1, 4) == UPSCALE_ERR_INVALID_ARGS, "row_begin < 0");
    CHECK(upscale_resample_rows(&s0, 2, 0, &d0, 5, 4) == UPSCALE_ERR_INVALID_ARGS, "begin > end");
    CHECK(upscale_resample_rows(&s0, 2, 0, &d0, 0, 9) == UPSCALE_ERR_INVALID_ARGS, "end > height");
    CHECK(upscale_resample_rows(&s0, 2, 0, &d0, 3, 3) == UPSCALE_OK, "empty band is ok");
    CHECK(upscale_unsharp_mask(NULL, 1.0f, 1.0f, 0) == UPSCALE_ERR_INVALID_ARGS, "unsharp null");
    CHECK(upscale_postprocess(&d0, NULL) == UPSCALE_ERR_INVALID_ARGS, "postprocess null opts");

    ImageBuffer out;
    memset(&out, 0x5A, sizeof out);
    CHECK(create_image_buffer(0, 4, 4, &out) == UPSCALE_ERR_INVALID_ARGS && out.data == NULL, "create 0 width");
    CHECK(create_image_buffer(4, 4, 7, &out) == UPSCALE_ERR_UNSUPPORTED && out.data == NULL, "create 7 ch");
    CHECK(create_image_buffer(32768, 8193, 4, &out) == UPSCALE_ERR_TOO_LARGE && out.data == NULL, "create > 1GiB");
    CHECK(create_image_buffer(40000, 4, 4, &out) == UPSCALE_ERR_TOO_LARGE && out.data == NULL, "create dim limit");
    CHECK(create_image_buffer(4, 4, 4, NULL) == UPSCALE_ERR_INVALID_ARGS, "create null out");

    o.scale = 3;
    CHECK(upscale_image(&s0, &o, &out) == UPSCALE_ERR_UNSUPPORTED && out.data == NULL, "upscale_image bad scale leaves out empty");
    upscaler_options_default(&o);
    o.sharpen_amount = 99.0f;
    CHECK(upscale_image(&s0, &o, &out) == UPSCALE_ERR_INVALID_ARGS && out.data == NULL, "upscale_image bad sharpen frees dst");
    CHECK(upscale_image(&s0, &o, NULL) == UPSCALE_ERR_INVALID_ARGS, "upscale_image null out");

    CHECK(upscaler_error_string(UPSCALE_ERR_OOM) != NULL && upscaler_error_string(12345) != NULL, "error_string never NULL");
    CHECK(upscaler_version() == UPSCALER_VERSION_NUMBER, "version");
}

/* ============================================================================
 *  T6: premultiplied-alpha clamp
 * ==========================================================================*/
static void test_premultiplied(void)
{
    printf("[T6] premultiplied alpha clamp\n");
    for (int ch = 2; ch <= 4; ch += 2) {
        Exact s, d; exact_alloc(&s, 20, 20, ch, 0); exact_alloc(&d, 40, 40, ch, 0);
        fill_random(&s);                                     /* deliberately NOT valid premultiplied data */
        UpscaleOptions o; upscaler_options_default(&o);
        o.kernel = UPSCALE_KERNEL_LANCZOS3; o.sharpen_amount = 1.0f; o.flags = UPSCALE_FLAG_PREMULTIPLIED_ALPHA;
        CHECK(upscale_image_into(&s.v, &o, &d.v) == UPSCALE_OK, "rc");
        int viol = 0;
        for (int y = 0; y < 40; ++y)
            for (int x = 0; x < 40; ++x) {
                const uint8_t *p = px(&d, x, y);
                for (int c = 0; c < ch - 1; ++c) if (p[c] > p[ch - 1]) ++viol;
            }
        CHECK(viol == 0, "%d pixels with colour > alpha (ch %d)", viol, ch);
        exact_free(&s); exact_free(&d);
    }
}

/* ============================================================================
 *  T7: allocator -- alignment, stride, zero-fill, double free, churn
 * ==========================================================================*/
static void test_allocator(void)
{
    printf("[T7] allocator\n");
    for (int i = 0; i < 2000; ++i) {
        const int w = 1 + (int)(rnd() % 97), h = 1 + (int)(rnd() % 53), ch = 1 + (int)(rnd() % 4);
        ImageBuffer b;
        const int32_t rc = create_image_buffer(w, h, ch, &b);
        if (rc != UPSCALE_OK) { CHECK(0, "create failed rc=%d", rc); continue; }
        CHECK(((uintptr_t)b.data % 64u) == 0, "data not 64-byte aligned");
        CHECK(b.stride % 16 == 0 && b.stride >= w * ch && b.stride < w * ch + 16, "bad stride %d for row %d", b.stride, w * ch);
        int zero = 1;
        for (size_t k = 0; k < (size_t)b.stride * (size_t)h; ++k) if (b.data[k]) { zero = 0; break; }
        CHECK(zero, "buffer not zero-initialised");
        memset(b.data, 0xFF, (size_t)b.stride * (size_t)h);              /* touch every byte for Valgrind */
        free_image_buffer(&b);
        CHECK(b.data == NULL && b.width == 0, "struct not cleared by free");
        free_image_buffer(&b);                                           /* double free: no-op */
    }
    free_image_buffer(NULL);
    /* end-to-end through the owning API, repeated: leak detector fodder */
    for (int i = 0; i < 50; ++i) {
        Exact s; exact_alloc(&s, 31, 17, 3, 0); fill_random(&s);
        UpscaleOptions o; upscaler_options_default(&o); o.scale = (i & 1) ? 4 : 2;
        ImageBuffer out;
        CHECK(upscale_image(&s.v, &o, &out) == UPSCALE_OK, "upscale_image");
        CHECK(out.width == 31 * o.scale && out.height == 17 * o.scale && out.channels == 3, "dims");
        free_image_buffer(&out);
        exact_free(&s);
    }
}

/* ============================================================================
 *  T8: larger image, odd size, everything on (ring wrap-around exercised)
 * ==========================================================================*/
static void test_larger(void)
{
    printf("[T8] larger image vs. reference\n");
    const int w = 123, h = 97, scale = 4;
    Exact s, d; exact_alloc(&s, w, h, 3, 5); exact_alloc(&d, w * scale, h * scale, 3, 0);
    fill_random(&s);
    uint8_t *ref = (uint8_t *)malloc((size_t)w * scale * (size_t)h * scale * 3u);
    ref_upscale(&s, scale, UPSCALE_KERNEL_LANCZOS3, ref);
    CHECK(upscale_resample_rows(&s.v, scale, UPSCALE_KERNEL_LANCZOS3, &d.v, 0, h * scale) == UPSCALE_OK, "rc");
    int maxd = 0;
    for (int y = 0; y < h * scale; ++y)
        for (int i = 0; i < w * scale * 3; ++i) {
            const int diff = abs((int)px(&d, 0, y)[i] - (int)ref[(size_t)y * (size_t)w * scale * 3u + (size_t)i]);
            if (diff > maxd) maxd = diff;
        }
    CHECK(maxd <= 1, "max diff %d", maxd);
    free(ref); exact_free(&s); exact_free(&d);
}

int main(void)
{
    printf("upscaler engine self-test (v%d)\n", upscaler_version());
    test_against_reference();
    test_dc_preservation();
    test_banding();
    test_unsharp();
    test_invalid_args();
    test_premultiplied();
    test_allocator();
    test_larger();
    printf("\n%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
