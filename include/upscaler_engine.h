/* ============================================================================
 * upscaler_engine.h  --  Pure C11 image upscaler (Catmull-Rom / Lanczos-3 + USM)
 *
 * Design rules
 *   * Zero third-party dependencies. Only libc + libm.
 *   * No global or static mutable state  =>  every function is reentrant and
 *     safe to call concurrently on different buffers (or on disjoint row
 *     bands of the same destination, see upscale_resample_rows()).
 *   * ABI is C-only: fixed-width integers, plain structs, no enums in
 *     signatures, no callbacks, no varargs. Safe for Dart FFI / JNI / ctypes.
 *   * Every fallible function returns an int32_t status code. Nothing aborts.
 *
 * Pixel format
 *   Interleaved, 8 bits per channel, row-major, 1..4 channels
 *   (Gray, Gray+A, RGB, RGBA). Rows are `stride` BYTES apart (stride may be
 *   larger than width*channels). Only the first width*channels bytes of each
 *   row are ever read or written; padding bytes are never touched.
 *
 * Alpha
 *   Every channel is filtered independently. For images with transparency feed
 *   PREMULTIPLIED data and set UPSCALE_FLAG_PREMULTIPLIED_ALPHA so ringing
 *   cannot push colour above alpha. Unsharp mask never modifies the alpha
 *   channel of 2- and 4-channel images.
 * ==========================================================================*/
#ifndef UPSCALER_ENGINE_H
#define UPSCALER_ENGINE_H

#include <stdint.h>

#if defined(_WIN32)
#  ifdef UPSCALER_BUILD_SHARED
#    define UPSCALER_API __declspec(dllexport)
#  else
#    define UPSCALER_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define UPSCALER_API __attribute__((visibility("default")))
#else
#  define UPSCALER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- version ------------------------------------------------------------ */
#define UPSCALER_VERSION_MAJOR 1
#define UPSCALER_VERSION_MINOR 0
#define UPSCALER_VERSION_PATCH 0
#define UPSCALER_VERSION_NUMBER \
    (UPSCALER_VERSION_MAJOR * 10000 + UPSCALER_VERSION_MINOR * 100 + UPSCALER_VERSION_PATCH)

/* ---- status codes (all functions returning int32_t use these) ----------- */
#define UPSCALE_OK                 0
#define UPSCALE_ERR_INVALID_ARGS (-1) /* NULL pointer, bad size/stride/range  */
#define UPSCALE_ERR_OOM          (-2) /* allocation failed                    */
#define UPSCALE_ERR_UNSUPPORTED  (-3) /* scale/kernel/channel count not supported */
#define UPSCALE_ERR_TOO_LARGE    (-4) /* exceeds UPSCALE_MAX_* limits         */
#define UPSCALE_ERR_OVERLAP      (-5) /* source and destination memory overlap */

/* ---- options ------------------------------------------------------------ */
#define UPSCALE_KERNEL_CATMULL_ROM 0  /* 4 taps, fastest, mild ringing       */
#define UPSCALE_KERNEL_LANCZOS3    1  /* 6 taps, sharpest, slight ringing    */

#define UPSCALE_FLAG_PREMULTIPLIED_ALPHA 1 /* clamp colour <= alpha on output */

/* ---- hard limits (checked with 64-bit math, never overflow) ------------- */
#define UPSCALE_MAX_DIMENSION      32768          /* per side, input & output */
#define UPSCALE_MAX_BUFFER_BYTES   (1u << 30)     /* 1 GiB per buffer          */
#define UPSCALE_MAX_SHARPEN_AMOUNT 5.0f
#define UPSCALE_MIN_SHARPEN_RADIUS 0.3f           /* Gaussian sigma, in px     */
#define UPSCALE_MAX_SHARPEN_RADIUS 5.0f

/* A pixel buffer. Layout (LP64): data@0, width@8, height@12, channels@16,
 * stride@20, sizeof == 24.  (ILP32: 20 bytes.)  Mirror it exactly in Dart/JNI.
 *
 * Ownership: a buffer filled in by create_image_buffer()/upscale_image() is
 * OWNED by the caller and must be released with free_image_buffer(). A struct
 * that merely points at foreign memory (Android Bitmap, a Dart-owned block,
 * stb_image output...) is a NON-OWNING VIEW and must never be passed to
 * free_image_buffer().                                                       */
typedef struct ImageBuffer {
    uint8_t *data;     /* 64-byte aligned when created by create_image_buffer */
    int32_t  width;    /* pixels                                              */
    int32_t  height;   /* rows                                                */
    int32_t  channels; /* 1..4                                                */
    int32_t  stride;   /* bytes between row starts, >= width*channels         */
} ImageBuffer;

/* 24 bytes, no padding. */
typedef struct UpscaleOptions {
    int32_t scale;             /* 2 or 4                                      */
    int32_t kernel;            /* UPSCALE_KERNEL_*                            */
    int32_t flags;             /* UPSCALE_FLAG_* bit mask                     */
    float   sharpen_amount;    /* 0 = off; typical 0.4..1.2; max 5            */
    float   sharpen_radius;    /* Gaussian sigma in OUTPUT pixels, 0.3..5     */
    int32_t sharpen_threshold; /* 0..255; |orig-blur| <= thr is left alone    */
} UpscaleOptions;

/* ---- utility ------------------------------------------------------------ */
UPSCALER_API int32_t    upscaler_version(void);                 /* UPSCALER_VERSION_NUMBER */
UPSCALER_API const char *upscaler_error_string(int32_t code);   /* static string, never NULL */
UPSCALER_API void       upscaler_options_default(UpscaleOptions *opts);

/* ---- memory ------------------------------------------------------------- */

/* Allocates a zero-filled width x height x channels buffer.
 * stride = align_up(width*channels, 16); data is 64-byte aligned.
 * `out` is ALWAYS overwritten: on failure out->data == NULL.
 * Always release with free_image_buffer() (never free()/delete/calloc.free).  */
UPSCALER_API int32_t create_image_buffer(int32_t width, int32_t height,
                                         int32_t channels, ImageBuffer *out);

/* Releases a buffer made by create_image_buffer()/upscale_image() and zeroes
 * the struct. NULL or an already-freed struct is a safe no-op.               */
UPSCALER_API void free_image_buffer(ImageBuffer *buf);

/* ---- upscaling ---------------------------------------------------------- */

/* Convenience: allocates *out_dst (width*scale x height*scale, same channel
 * count), upscales, post-processes. On error *out_dst is left empty (data NULL)
 * and nothing leaks.                                                         */
UPSCALER_API int32_t upscale_image(const ImageBuffer *src,
                                   const UpscaleOptions *opts,
                                   ImageBuffer *out_dst);

/* Zero-allocation-of-pixels variant: dst is caller-provided memory with
 * dst->width == src->width*scale, dst->height == src->height*scale and
 * dst->channels == src->channels (dst may be a view onto any memory).        */
UPSCALER_API int32_t upscale_image_into(const ImageBuffer *src,
                                        const UpscaleOptions *opts,
                                        const ImageBuffer *dst);

/* ---- building blocks (for multi-threaded callers) ------------------------
 * Pass 1 -- resample destination rows [row_begin, row_end). Bands are fully
 * independent and bit-identical to a single full-image call, so N threads /
 * Dart isolates may each take a band of the SAME dst concurrently.
 * Pass 2 -- upscale_postprocess() once, single-threaded, after all bands
 * are done (unsharp mask needs neighbouring rows).                           */
UPSCALER_API int32_t upscale_resample_rows(const ImageBuffer *src,
                                           int32_t scale, int32_t kernel,
                                           const ImageBuffer *dst,
                                           int32_t row_begin, int32_t row_end);

UPSCALER_API int32_t upscale_postprocess(const ImageBuffer *img,
                                         const UpscaleOptions *opts);

/* In-place unsharp mask: out = orig + amount * (orig - gaussian_blur(orig)),
 * clamped to 0..255. Colour channels only. amount == 0 is a no-op.          */
UPSCALER_API int32_t upscale_unsharp_mask(const ImageBuffer *img, float amount,
                                          float radius, int32_t threshold);

#ifdef __cplusplus
}
#endif
#endif /* UPSCALER_ENGINE_H */
