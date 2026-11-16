/* ============================================================================
 * upscaler_engine.c  --  Pure C11 separable polyphase upscaler + unsharp mask
 *
 * ALGORITHM IN ONE PARAGRAPH
 *   Output pixel x of an integer-scale-s upscale samples the source at
 *       sx = (x + 0.5)/s - 0.5                     (pixel-centre convention)
 *   Writing x = q*s + r (0 <= r < s) gives sx = q + (2r+1-s)/(2s): the
 *   FRACTIONAL PART ONLY DEPENDS ON r. Hence an s-x upscale needs just `s`
 *   distinct filters ("phases"), not one per pixel. We evaluate the kernel
 *   (Catmull-Rom: 4 taps, Lanczos-3: 6 taps) once per phase, quantise to Q14
 *   int16 weights that sum to EXACTLY 16384, and from then on the hot loops are
 *   branch-free integer multiply-accumulates over uint8 data -- the exact shape
 *   ARM NEON (and the auto-vectoriser) likes: u8 x s16 -> s32 widening MACs.
 *
 * MEMORY MATH (why it is cache friendly)
 *   A naive separable resize materialises the whole horizontally-resampled
 *   intermediate: dst_w * src_h * ch bytes. For 1920x1080 RGBA at 2x that is
 *   3840*1080*4 = 16.6 MB -- far larger than any CPU cache. Instead we stream:
 *     for each source row: pad edges -> horizontal pass -> ring slot (row & 7)
 *     for each output row:  vertical pass over the <= 6 ring rows it needs
 *   Scratch = padded row + 8 ring rows = ~(W+6)*ch + 8*align16(W*s*ch) bytes
 *   -> 1920x1080 RGBA 2x: ~15 KB + 8*15 KB = ~135 KB (fits in L2, ~120x less).
 *   Source rows are read once, strictly sequentially (prefetcher friendly).
 *
 * EDGE HANDLING (no out-of-bounds reads, ever)
 *   Horizontal: each source row is copied into a scratch row padded with
 *   taps/2 replicated edge pixels on both sides, so the inner loop needs no
 *   clamping at all. Vertical: row indices are clamped to [0, h-1] before
 *   selecting ring slots. => "clamp-to-edge" everywhere, branch-free hot loops.
 *
 * FIXED POINT
 *   Weights Q14 (int16). acc = 8192 + sum(w*px) stays far inside int32
 *   (worst case |sum| < 255 * 1.5 * 16384 ~ 6.3e6). Result = clamp(acc >> 14).
 *   Weight sums are exactly 16384 so flat regions are reproduced bit-exactly.
 *   Needs arithmetic >> on negative ints (static-asserted below; true on every
 *   compiler/ABI Android targets).
 * ==========================================================================*/
#include "upscaler_engine.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(CHAR_BIT == 8, "8-bit bytes required");
_Static_assert((-1 >> 1) == -1, "arithmetic right shift of negative ints required");
_Static_assert(sizeof(UpscaleOptions) == 24, "UpscaleOptions ABI size changed");
_Static_assert(offsetof(ImageBuffer, width) == sizeof(void *), "ImageBuffer ABI layout changed");
_Static_assert(sizeof(int32_t) == 4, "int32_t must be 4 bytes");

/* ---- tuning constants ---------------------------------------------------- */
#define FP_SHIFT       14
#define FP_ONE         (1 << FP_SHIFT)
#define FP_HALF        (1 << (FP_SHIFT - 1))
#define MAX_TAPS       6          /* Lanczos-3                                  */
#define MAX_PHASES     4          /* scale 4                                    */
#define RING_ROWS      8          /* power of two >= MAX_TAPS                   */
#define MAX_BLUR_R     15         /* sigma <= 5  ->  R = ceil(3*sigma) <= 15    */
#define MAX_BLUR_TAPS  (2 * MAX_BLUR_R + 1)
#define BLOCK          256        /* accumulator block: 1 KiB, lives in L1      */
#define BUF_ALIGN      64         /* cache line                                 */
#define ROW_ALIGN      16         /* 128-bit NEON register                      */
#define PI_D           3.14159265358979323846

/* ---- tiny helpers -------------------------------------------------------- */
static inline size_t align_up_sz(size_t v, size_t a)      /* a: power of two */
{
    return (v + a - 1u) & ~(a - 1u);
}
static inline int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}
static inline uint8_t clamp_u8(int32_t v)
{
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

/* Bytes between first and last byte touched in a buffer. */
static uint64_t span_bytes(const ImageBuffer *b)
{
    return (uint64_t)(b->height - 1) * (uint64_t)b->stride +
           (uint64_t)b->width * (uint64_t)b->channels;
}

/* ============================================================================
 *  Validation
 * ==========================================================================*/
static int32_t check_buffer(const ImageBuffer *b)
{
    if (b == NULL || b->data == NULL) return UPSCALE_ERR_INVALID_ARGS;
    if (b->channels < 1 || b->channels > 4) return UPSCALE_ERR_UNSUPPORTED;
    if (b->width < 1 || b->height < 1) return UPSCALE_ERR_INVALID_ARGS;
    if (b->width > UPSCALE_MAX_DIMENSION || b->height > UPSCALE_MAX_DIMENSION)
        return UPSCALE_ERR_TOO_LARGE;
    if ((int64_t)b->stride < (int64_t)b->width * b->channels) return UPSCALE_ERR_INVALID_ARGS;
    if (span_bytes(b) > (uint64_t)UPSCALE_MAX_BUFFER_BYTES) return UPSCALE_ERR_TOO_LARGE;
    return UPSCALE_OK;
}

static int32_t check_sharpen_params(float amount, float radius, int32_t threshold)
{
    /* Written as !(a >= lo && a <= hi) so NaN is rejected too. */
    if (!(amount >= 0.0f && amount <= UPSCALE_MAX_SHARPEN_AMOUNT)) return UPSCALE_ERR_INVALID_ARGS;
    if (!(radius >= UPSCALE_MIN_SHARPEN_RADIUS && radius <= UPSCALE_MAX_SHARPEN_RADIUS))
        return UPSCALE_ERR_INVALID_ARGS;
    if (threshold < 0 || threshold > 255) return UPSCALE_ERR_INVALID_ARGS;
    return UPSCALE_OK;
}

static int32_t check_post_options(const UpscaleOptions *o)
{
    if (o == NULL) return UPSCALE_ERR_INVALID_ARGS;
    if ((o->flags & ~UPSCALE_FLAG_PREMULTIPLIED_ALPHA) != 0) return UPSCALE_ERR_INVALID_ARGS;
    return check_sharpen_params(o->sharpen_amount, o->sharpen_radius, o->sharpen_threshold);
}

static int ranges_overlap(const void *a, uint64_t alen, const void *b, uint64_t blen)
{
    const uintptr_t a0 = (uintptr_t)a, b0 = (uintptr_t)b;
    return (uint64_t)a0 < (uint64_t)b0 + blen && (uint64_t)b0 < (uint64_t)a0 + alen;
}

static int32_t check_geometry(const ImageBuffer *src, int32_t scale, int32_t kernel,
                              const ImageBuffer *dst)
{
    int32_t rc = check_buffer(src);
    if (rc != UPSCALE_OK) return rc;
    rc = check_buffer(dst);
    if (rc != UPSCALE_OK) return rc;
    if (scale != 2 && scale != 4) return UPSCALE_ERR_UNSUPPORTED;
    if (kernel != UPSCALE_KERNEL_CATMULL_ROM && kernel != UPSCALE_KERNEL_LANCZOS3)
        return UPSCALE_ERR_UNSUPPORTED;
    if (src->channels != dst->channels) return UPSCALE_ERR_INVALID_ARGS;
    if ((int64_t)src->width * scale != dst->width || (int64_t)src->height * scale != dst->height)
        return UPSCALE_ERR_INVALID_ARGS;
    if (ranges_overlap(src->data, span_bytes(src), dst->data, span_bytes(dst)))
        return UPSCALE_ERR_OVERLAP;
    return UPSCALE_OK;
}

/* ============================================================================
 *  Utility API
 * ==========================================================================*/
int32_t upscaler_version(void) { return UPSCALER_VERSION_NUMBER; }

const char *upscaler_error_string(int32_t code)
{
    switch (code) {
    case UPSCALE_OK:                 return "ok";
    case UPSCALE_ERR_INVALID_ARGS:   return "invalid arguments";
    case UPSCALE_ERR_OOM:            return "out of memory";
    case UPSCALE_ERR_UNSUPPORTED:    return "unsupported scale, kernel or channel count";
    case UPSCALE_ERR_TOO_LARGE:      return "image too large";
    case UPSCALE_ERR_OVERLAP:        return "source and destination overlap";
    default:                         return "unknown error";
    }
}

void upscaler_options_default(UpscaleOptions *opts)
{
    if (opts == NULL) return;
    opts->scale             = 2;
    opts->kernel            = UPSCALE_KERNEL_LANCZOS3;
    opts->flags             = 0;
    opts->sharpen_amount    = 0.6f;
    opts->sharpen_radius    = 1.0f;
    opts->sharpen_threshold = 0;
}

/* ============================================================================
 *  Memory
 *
 *  Layout of one allocation (malloc'd block, 64-byte aligned payload):
 *
 *   raw                          data (64-aligned)
 *    |<-- padding -->|<- void* raw ->|<---------- stride*height bytes ------>|
 *                     ^ stored at data - sizeof(void*)
 *
 *  Over-allocation = BUF_ALIGN + sizeof(void*) bytes. free_image_buffer()
 *  recovers `raw` from the hidden slot, so no extra struct field is needed and
 *  the public ABI stays minimal. (aligned_alloc is avoided on purpose: it is
 *  missing from older Android API levels.)
 * ==========================================================================*/
int32_t create_image_buffer(int32_t width, int32_t height, int32_t channels, ImageBuffer *out)
{
    if (out == NULL) return UPSCALE_ERR_INVALID_ARGS;
    memset(out, 0, sizeof *out);

    if (channels < 1 || channels > 4) return UPSCALE_ERR_UNSUPPORTED;
    if (width < 1 || height < 1) return UPSCALE_ERR_INVALID_ARGS;
    if (width > UPSCALE_MAX_DIMENSION || height > UPSCALE_MAX_DIMENSION) return UPSCALE_ERR_TOO_LARGE;

    const uint64_t row_bytes = (uint64_t)width * (uint64_t)channels;          /* <= 131072 */
    const uint64_t stride    = (row_bytes + (ROW_ALIGN - 1u)) & ~(uint64_t)(ROW_ALIGN - 1u);
    const uint64_t total     = stride * (uint64_t)height;                     /* <= 2^32.. */
    if (total > (uint64_t)UPSCALE_MAX_BUFFER_BYTES) return UPSCALE_ERR_TOO_LARGE;
    if (total + BUF_ALIGN + sizeof(void *) > (uint64_t)SIZE_MAX) return UPSCALE_ERR_TOO_LARGE;

    void *raw = malloc((size_t)total + BUF_ALIGN + sizeof(void *));
    if (raw == NULL) return UPSCALE_ERR_OOM;

    uintptr_t p = ((uintptr_t)raw + sizeof(void *) + (BUF_ALIGN - 1u)) & ~(uintptr_t)(BUF_ALIGN - 1u);
    ((void **)p)[-1] = raw;                         /* hidden back-pointer     */
    memset((void *)p, 0, (size_t)total);            /* deterministic contents  */

    out->data     = (uint8_t *)p;
    out->width    = width;
    out->height   = height;
    out->channels = channels;
    out->stride   = (int32_t)stride;
    return UPSCALE_OK;
}

void free_image_buffer(ImageBuffer *buf)
{
    if (buf == NULL) return;
    if (buf->data != NULL) free(((void **)buf->data)[-1]);
    memset(buf, 0, sizeof *buf);                    /* double free => no-op    */
}

/* ============================================================================
 *  Kernels and polyphase weight bank
 * ==========================================================================*/
static double kernel_catmull_rom(double x)          /* cubic convolution, a=-0.5 */
{
    x = fabs(x);
    if (x < 1.0) return (1.5 * x - 2.5) * x * x + 1.0;
    if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    return 0.0;
}

static double sinc_pi(double x)
{
    if (fabs(x) < 1e-12) return 1.0;
    const double px = PI_D * x;
    return sin(px) / px;
}

static double kernel_lanczos3(double x)
{
    x = fabs(x);
    return x < 3.0 ? sinc_pi(x) * sinc_pi(x / 3.0) : 0.0;
}

/* Normalise `n` real weights to Q14 so that sum == FP_ONE exactly: round each,
 * then dump the rounding residue on the largest-magnitude tap.               */
static void quantise_weights(const double *wd, int n, int16_t *out)
{
    double sum = 0.0;
    for (int t = 0; t < n; ++t) sum += wd[t];

    int32_t q[MAX_BLUR_TAPS];
    int32_t total = 0;
    int best = 0;
    for (int t = 0; t < n; ++t) {
        q[t] = (int32_t)floor(wd[t] / sum * (double)FP_ONE + 0.5);
        total += q[t];
        if (abs(q[t]) > abs(q[best])) best = t;
    }
    q[best] += FP_ONE - total;
    for (int t = 0; t < n; ++t) out[t] = (int16_t)q[t];
}

typedef struct PolyBank {
    int32_t taps;                         /* 4 or 6                            */
    int32_t scale;                        /* 2 or 4 (= number of phases)       */
    int32_t off[MAX_PHASES];              /* floor(sx) - q : 0 or -1           */
    int32_t start[MAX_PHASES];            /* index into padded row, relative to q */
    int16_t w[MAX_PHASES][MAX_TAPS];      /* Q14, each row sums to 16384       */
} PolyBank;

static void build_bank(int32_t kernel, int32_t scale, PolyBank *bank)
{
    const int32_t taps    = (kernel == UPSCALE_KERNEL_LANCZOS3) ? 6 : 4;
    const int32_t left    = taps / 2 - 1;     /* taps strictly left of floor(sx) */
    const int32_t pad_px  = taps / 2;
    memset(bank, 0, sizeof *bank);
    bank->taps  = taps;
    bank->scale = scale;

    for (int32_t r = 0; r < scale; ++r) {
        const int32_t num = 2 * r + 1 - scale;          /* sx - q = num / (2s) */
        double frac;
        if (num < 0) { bank->off[r] = -1; frac = (double)(num + 2 * scale) / (2.0 * scale); }
        else         { bank->off[r] =  0; frac = (double)num / (2.0 * scale); }
        bank->start[r] = bank->off[r] - left + pad_px;  /* >= 0 by construction */

        double wd[MAX_TAPS];
        for (int32_t t = 0; t < taps; ++t) {
            const double d = frac - (double)(t - left); /* sx - tap position   */
            wd[t] = (kernel == UPSCALE_KERNEL_LANCZOS3) ? kernel_lanczos3(d) : kernel_catmull_rom(d);
        }
        quantise_weights(wd, taps, bank->w[r]);
    }
}

/* ============================================================================
 *  Row primitives
 * ==========================================================================*/

/* Copy one source row into `pad`, replicating the edge pixels pad_px times on
 * both sides. After this, pixel index p in `pad` == source pixel p - pad_px,
 * and every p in [0, w + 2*pad_px) is valid: no clamping needed afterwards.  */
static void load_padded_row(uint8_t *pad, const uint8_t *srow, int32_t w, int32_t ch, int32_t pad_px)
{
    const size_t c = (size_t)ch;
    memcpy(pad + (size_t)pad_px * c, srow, (size_t)w * c);
    for (int32_t i = 0; i < pad_px; ++i) {
        memcpy(pad + (size_t)i * c, srow, c);                                   /* left  */
        memcpy(pad + (size_t)(pad_px + w + i) * c, srow + (size_t)(w - 1) * c, c); /* right */
    }
}

/* Horizontal polyphase pass for ONE row: w source pixels -> w*scale pixels.
 * `taps` and `ch` are literal constants at every call site below, so after
 * inlining the compiler fully unrolls the tap/channel loops.                 */
static inline void hresample_impl(const uint8_t *restrict pad, uint8_t *restrict out,
                                  int32_t w, const PolyBank *bank, const int32_t taps,
                                  const int32_t ch)
{
    const int32_t scale = bank->scale;
    for (int32_t q = 0; q < w; ++q) {
        for (int32_t r = 0; r < scale; ++r) {
            const int16_t *k = bank->w[r];
            const uint8_t *p = pad + (size_t)(q + bank->start[r]) * (size_t)ch;
            uint8_t *o = out + ((size_t)q * (size_t)scale + (size_t)r) * (size_t)ch;
            for (int32_t c = 0; c < ch; ++c) {
                int32_t acc = FP_HALF;
                for (int32_t t = 0; t < taps; ++t) acc += (int32_t)k[t] * (int32_t)p[t * ch + c];
                o[c] = clamp_u8(acc >> FP_SHIFT);
            }
        }
    }
}

static void hresample_row(const uint8_t *pad, uint8_t *out, int32_t w, const PolyBank *bank, int32_t ch)
{
    if (bank->taps == 4) {
        switch (ch) {
        case 1:  hresample_impl(pad, out, w, bank, 4, 1);  break;
        case 3:  hresample_impl(pad, out, w, bank, 4, 3);  break;
        case 4:  hresample_impl(pad, out, w, bank, 4, 4);  break;
        default: hresample_impl(pad, out, w, bank, 4, ch); break;
        }
    } else {
        switch (ch) {
        case 1:  hresample_impl(pad, out, w, bank, 6, 1);  break;
        case 3:  hresample_impl(pad, out, w, bank, 6, 3);  break;
        case 4:  hresample_impl(pad, out, w, bank, 6, 4);  break;
        default: hresample_impl(pad, out, w, bank, 6, ch); break;
        }
    }
}

/* Vertical pass: out[i] = clamp(sum_t k[t] * rows[t][i]) over n contiguous
 * bytes. This is THE hot loop (stride-1 loads, one weight per row, no
 * branches) and maps 1:1 onto NEON  vmull/vmlal (u8->s16 x s16 -> s32).
 * Row pointers may alias each other (clamped edge rows) -- they are only read. */
static void vresample_row(uint8_t *restrict out, const uint8_t *const *rows,
                          const int16_t *k, int32_t taps, size_t n)
{
    if (taps == 4) {
        const uint8_t *restrict r0 = rows[0], *restrict r1 = rows[1];
        const uint8_t *restrict r2 = rows[2], *restrict r3 = rows[3];
        const int32_t k0 = k[0], k1 = k[1], k2 = k[2], k3 = k[3];
        for (size_t i = 0; i < n; ++i) {
            const int32_t acc = FP_HALF + k0 * r0[i] + k1 * r1[i] + k2 * r2[i] + k3 * r3[i];
            out[i] = clamp_u8(acc >> FP_SHIFT);
        }
    } else {
        const uint8_t *restrict r0 = rows[0], *restrict r1 = rows[1], *restrict r2 = rows[2];
        const uint8_t *restrict r3 = rows[3], *restrict r4 = rows[4], *restrict r5 = rows[5];
        const int32_t k0 = k[0], k1 = k[1], k2 = k[2], k3 = k[3], k4 = k[4], k5 = k[5];
        for (size_t i = 0; i < n; ++i) {
            const int32_t acc = FP_HALF + k0 * r0[i] + k1 * r1[i] + k2 * r2[i] +
                                k3 * r3[i] + k4 * r4[i] + k5 * r5[i];
            out[i] = clamp_u8(acc >> FP_SHIFT);
        }
    }
}

/* ============================================================================
 *  Resampler core (streaming, ring-buffered)
 * ==========================================================================*/
static int32_t resample_rows_core(const ImageBuffer *src, int32_t scale, int32_t kernel,
                                  const ImageBuffer *dst, int32_t y_begin, int32_t y_end)
{
    PolyBank bank;
    build_bank(kernel, scale, &bank);

    const int32_t taps   = bank.taps;
    const int32_t left   = taps / 2 - 1;
    const int32_t pad_px = taps / 2;
    const int32_t w = src->width, h = src->height, ch = src->channels;

    const size_t dst_row_bytes = (size_t)dst->width * (size_t)ch;
    const size_t ring_stride   = align_up_sz(dst_row_bytes, ROW_ALIGN);
    const size_t pad_bytes     = align_up_sz((size_t)(w + 2 * pad_px) * (size_t)ch, ROW_ALIGN);

    /* One allocation, freed on the single exit path below. */
    uint8_t *scratch = (uint8_t *)malloc(pad_bytes + (size_t)RING_ROWS * ring_stride);
    if (scratch == NULL) return UPSCALE_ERR_OOM;
    uint8_t *pad  = scratch;
    uint8_t *ring = scratch + pad_bytes;

    /* First source row this band needs (rows below it are never touched). */
    int32_t next_row;
    {
        const int32_t b0 = y_begin / scale + bank.off[y_begin % scale];
        next_row = clamp_i32(b0 - left, 0, h - 1);
    }

    for (int32_t y = y_begin; y < y_end; ++y) {
        const int32_t qy = y / scale, ry = y % scale;
        const int32_t b  = qy + bank.off[ry];                    /* floor(sy)            */
        const int32_t hi = clamp_i32(b + taps / 2, 0, h - 1);    /* last source row used */

        /* Horizontally resample every source row up to `hi` exactly once. */
        while (next_row <= hi) {
            load_padded_row(pad, src->data + (size_t)next_row * (size_t)src->stride, w, ch, pad_px);
            hresample_row(pad, ring + (size_t)(next_row & (RING_ROWS - 1)) * ring_stride, w, &bank, ch);
            ++next_row;
        }

        /* Rows b-left .. b+taps/2 (clamped) are all inside the last RING_ROWS
         * computed rows because (taps-1) < RING_ROWS and clamping is monotone. */
        const uint8_t *rows[MAX_TAPS];
        for (int32_t t = 0; t < taps; ++t) {
            const int32_t sy = clamp_i32(b - left + t, 0, h - 1);
            rows[t] = ring + (size_t)(sy & (RING_ROWS - 1)) * ring_stride;
        }
        vresample_row(dst->data + (size_t)y * (size_t)dst->stride, rows, bank.w[ry], taps, dst_row_bytes);
    }

    free(scratch);
    return UPSCALE_OK;
}

/* ============================================================================
 *  Unsharp mask (separable Gaussian, streamed, in place)
 *
 *  scratch = padded row + ring_rows (pow2 >= 2R+1, <= 32) blurred rows + 1 row.
 *  Row y of the image is overwritten only AFTER every ring row that depends on
 *  it was produced (ring row r is built when processing y = r - R, and rows
 *  are modified strictly in increasing order), so in-place is exact.
 * ==========================================================================*/

/* Symmetric 1-D filter over `n` contiguous bytes:
 *     out[i] = clamp((HALF + sum_{k=0..2R} g[k] * rows[k][i]) >> 14)
 * The Gaussian is symmetric (g[k] == g[2R-k], preserved by quantise_weights because the
 * rounding residue lands on the unique centre tap), so mirrored taps are ADDED first and
 * multiplied once: R+1 multiplies per output instead of 2R+1. Two tap-pairs are handled
 * per sweep so the int32 accumulator block (1 KiB, L1-resident) is touched ~R/2 times
 * instead of 2R+1 times -- this loop, not the arithmetic, is what bounds the speed.
 *
 *   Vertical   : rows[k] = ring row for image row y-R+k
 *   Horizontal : rows[k] = padded_row + k*channels  (a shifted view of ONE row)
 * so a single routine serves both directions.                                   */
static void filter_sym(uint8_t *restrict out, const uint8_t *const *rows, size_t n,
                       const int16_t *g, int32_t R)
{
    int32_t acc[BLOCK];
    for (size_t i0 = 0; i0 < n; i0 += BLOCK) {
        const size_t len = (n - i0 < BLOCK) ? (n - i0) : BLOCK;

        const uint8_t *c = rows[R] + i0;
        const int32_t gc = g[R];
        for (size_t i = 0; i < len; ++i) acc[i] = FP_HALF + gc * (int32_t)c[i];

        int32_t k = 0;
        for (; k + 1 < R; k += 2) {
            const uint8_t *a0 = rows[k] + i0,     *b0 = rows[2 * R - k] + i0;
            const uint8_t *a1 = rows[k + 1] + i0, *b1 = rows[2 * R - k - 1] + i0;
            const int32_t g0 = g[k], g1 = g[k + 1];
            for (size_t i = 0; i < len; ++i)
                acc[i] += g0 * ((int32_t)a0[i] + (int32_t)b0[i]) + g1 * ((int32_t)a1[i] + (int32_t)b1[i]);
        }
        if (k < R) {
            const uint8_t *a0 = rows[k] + i0, *b0 = rows[2 * R - k] + i0;
            const int32_t g0 = g[k];
            for (size_t i = 0; i < len; ++i) acc[i] += g0 * ((int32_t)a0[i] + (int32_t)b0[i]);
        }
        for (size_t i = 0; i < len; ++i) out[i0 + i] = clamp_u8(acc[i] >> FP_SHIFT);
    }
}

/* Branch-free unsharp step over a whole row (vectorises; the data-dependent threshold
 * test becomes a mask):   blur[i] <- clamp(orig[i] + (|orig-blur| > thr ? (orig-blur)*amount : 0)). */
static void usm_combine(uint8_t *restrict blur_io, const uint8_t *restrict orig, size_t n,
                        int32_t amt_q8, int32_t thr)
{
    for (size_t i = 0; i < n; ++i) {
        const int32_t o     = orig[i];
        const int32_t d     = o - (int32_t)blur_io[i];
        const int32_t delta = (d * amt_q8 + 128) >> 8;
        const int32_t keep  = -(int32_t)((d > thr) | (d < -thr));   /* 0 or all-ones */
        blur_io[i] = clamp_u8(o + (delta & keep));
    }
}

int32_t upscale_unsharp_mask(const ImageBuffer *img, float amount, float radius, int32_t threshold)
{
    int32_t rc = check_buffer(img);
    if (rc != UPSCALE_OK) return rc;
    rc = check_sharpen_params(amount, radius, threshold);
    if (rc != UPSCALE_OK) return rc;
    if (amount == 0.0f) return UPSCALE_OK;

    const int32_t w = img->width, h = img->height, ch = img->channels;
    const int32_t ncolor = (ch == 2 || ch == 4) ? ch - 1 : ch;     /* keep alpha */
    const int32_t amt_q8 = (int32_t)(amount * 256.0f + 0.5f);

    /* Gaussian kernel, R = ceil(3 sigma) in [1,15], Q14 weights summing to 16384. */
    int32_t R = (int32_t)ceil(3.0 * (double)radius);
    R = clamp_i32(R, 1, MAX_BLUR_R);
    const int32_t ntaps = 2 * R + 1;
    int16_t g[MAX_BLUR_TAPS];
    {
        double wd[MAX_BLUR_TAPS];
        const double sigma = (double)radius;
        for (int32_t i = 0; i < ntaps; ++i) {
            const double d = (double)(i - R);
            wd[i] = exp(-0.5 * d * d / (sigma * sigma));
        }
        quantise_weights(wd, ntaps, g);
    }

    int32_t ring_rows = 1;
    while (ring_rows < ntaps) ring_rows <<= 1;                      /* <= 32 */
    const int32_t ring_mask = ring_rows - 1;

    const size_t row_bytes  = (size_t)w * (size_t)ch;
    const size_t row_stride = align_up_sz(row_bytes, ROW_ALIGN);
    const size_t pad_bytes  = align_up_sz((size_t)(w + 2 * R) * (size_t)ch, ROW_ALIGN);

    uint8_t *scratch = (uint8_t *)malloc(pad_bytes + ((size_t)ring_rows + 1u) * row_stride);
    if (scratch == NULL) return UPSCALE_ERR_OOM;
    uint8_t *pad  = scratch;
    uint8_t *ring = scratch + pad_bytes;
    uint8_t *vrow = ring + (size_t)ring_rows * row_stride;

    /* Horizontal taps are shifted views of the one padded row: tap k starts k pixels in. */
    const uint8_t *hrows[MAX_BLUR_TAPS];
    for (int32_t k = 0; k < ntaps; ++k) hrows[k] = pad + (size_t)k * (size_t)ch;

    int32_t next = 0;                                               /* next row to h-blur */
    for (int32_t y = 0; y < h; ++y) {
        const int32_t hi = (y + R < h - 1) ? (y + R) : (h - 1);
        while (next <= hi) {
            load_padded_row(pad, img->data + (size_t)next * (size_t)img->stride, w, ch, R);
            filter_sym(ring + (size_t)(next & ring_mask) * row_stride, hrows, row_bytes, g, R);
            ++next;
        }

        const uint8_t *rows[MAX_BLUR_TAPS];
        for (int32_t k = 0; k < ntaps; ++k) {
            const int32_t sy = clamp_i32(y - R + k, 0, h - 1);
            rows[k] = ring + (size_t)(sy & ring_mask) * row_stride;
        }
        filter_sym(vrow, rows, row_bytes, g, R);                    /* vrow = blur(row y) */

        uint8_t *o = img->data + (size_t)y * (size_t)img->stride;   /* original row y */
        usm_combine(vrow, o, row_bytes, amt_q8, threshold);         /* vrow = sharpened row y */
        if (ncolor == ch) {
            memcpy(o, vrow, row_bytes);
        } else {                                                    /* leave the alpha byte alone */
            for (int32_t x = 0; x < w; ++x)
                for (int32_t c = 0; c < ncolor; ++c)
                    o[(size_t)x * (size_t)ch + (size_t)c] = vrow[(size_t)x * (size_t)ch + (size_t)c];
        }
    }

    free(scratch);
    return UPSCALE_OK;
}

/* Premultiplied images: colour must never exceed alpha (Lanczos ringing and
 * sharpening can violate that and cause bright fringes when composited).    */
static void clamp_color_to_alpha(const ImageBuffer *img)
{
    const int32_t ch = img->channels;
    if (ch != 2 && ch != 4) return;
    for (int32_t y = 0; y < img->height; ++y) {
        uint8_t *o = img->data + (size_t)y * (size_t)img->stride;
        for (int32_t x = 0; x < img->width; ++x, o += ch) {
            const uint8_t a = o[ch - 1];
            for (int32_t c = 0; c < ch - 1; ++c)
                if (o[c] > a) o[c] = a;
        }
    }
}

/* ============================================================================
 *  Public upscaling API
 * ==========================================================================*/
int32_t upscale_resample_rows(const ImageBuffer *src, int32_t scale, int32_t kernel,
                              const ImageBuffer *dst, int32_t row_begin, int32_t row_end)
{
    const int32_t rc = check_geometry(src, scale, kernel, dst);
    if (rc != UPSCALE_OK) return rc;
    if (row_begin < 0 || row_end > dst->height || row_begin > row_end) return UPSCALE_ERR_INVALID_ARGS;
    if (row_begin == row_end) return UPSCALE_OK;
    return resample_rows_core(src, scale, kernel, dst, row_begin, row_end);
}

int32_t upscale_postprocess(const ImageBuffer *img, const UpscaleOptions *opts)
{
    int32_t rc = check_buffer(img);
    if (rc != UPSCALE_OK) return rc;
    rc = check_post_options(opts);
    if (rc != UPSCALE_OK) return rc;

    if (opts->sharpen_amount > 0.0f) {
        rc = upscale_unsharp_mask(img, opts->sharpen_amount, opts->sharpen_radius,
                                  opts->sharpen_threshold);
        if (rc != UPSCALE_OK) return rc;
    }
    if (opts->flags & UPSCALE_FLAG_PREMULTIPLIED_ALPHA) clamp_color_to_alpha(img);
    return UPSCALE_OK;
}

int32_t upscale_image_into(const ImageBuffer *src, const UpscaleOptions *opts, const ImageBuffer *dst)
{
    if (opts == NULL) return UPSCALE_ERR_INVALID_ARGS;
    int32_t rc = check_geometry(src, opts->scale, opts->kernel, dst);
    if (rc != UPSCALE_OK) return rc;
    rc = check_post_options(opts);                 /* validate BEFORE any pixel work */
    if (rc != UPSCALE_OK) return rc;

    rc = resample_rows_core(src, opts->scale, opts->kernel, dst, 0, dst->height);
    if (rc != UPSCALE_OK) return rc;
    return upscale_postprocess(dst, opts);
}

int32_t upscale_image(const ImageBuffer *src, const UpscaleOptions *opts, ImageBuffer *out_dst)
{
    if (out_dst == NULL || src == out_dst) return UPSCALE_ERR_INVALID_ARGS;
    memset(out_dst, 0, sizeof *out_dst);
    if (src == NULL || opts == NULL) return UPSCALE_ERR_INVALID_ARGS;

    int32_t rc = check_buffer(src);
    if (rc != UPSCALE_OK) return rc;
    if (opts->scale != 2 && opts->scale != 4) return UPSCALE_ERR_UNSUPPORTED;

    ImageBuffer tmp;
    rc = create_image_buffer(src->width * opts->scale, src->height * opts->scale, src->channels, &tmp);
    if (rc != UPSCALE_OK) return rc;

    rc = upscale_image_into(src, opts, &tmp);
    if (rc != UPSCALE_OK) {
        free_image_buffer(&tmp);                   /* no leak on any failure path */
        return rc;
    }
    *out_dst = tmp;
    return UPSCALE_OK;
}
