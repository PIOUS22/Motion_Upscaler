/* ============================================================================
 * main.c -- desktop test runner for the upscaler engine
 *
 *   upscaler_cli photo.png out.png --scale 2 --kernel lanczos --sharpen 0.6
 *   upscaler_cli synthetic:1280x720 out.png --scale 4 --bench 20
 *   upscaler_cli photo.png --eval --scale 2         (PSNR vs. box-downscaled)
 *
 * The harness only uses the PUBLIC API (upscaler_engine.h), exactly like the
 * Dart/JNI bridge will, plus stb_image / stb_image_write for file I/O.
 * ==========================================================================*/
#define _POSIX_C_SOURCE 200809L     /* clock_gettime under -std=c11 */

#include "upscaler_engine.h"
#include "stb_image.h"
#include "stb_image_write.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- timing -------------------------------------------------------------- */
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* ---- CLI ----------------------------------------------------------------- */
typedef struct Cli {
    const char    *in;
    const char    *out;
    UpscaleOptions opt;
    int            bench;   /* timed iterations, 0 = off */
    int            eval;
} Cli;

static void usage(const char *argv0)
{
    fprintf(stderr,
        "usage: %s <input.(png|jpg|bmp...)|synthetic:WxH> [output.(png|jpg|bmp)] [options]\n"
        "  --scale 2|4            upscale factor (default 2)\n"
        "  --kernel catmull|lanczos   resampling kernel (default lanczos)\n"
        "  --sharpen A            unsharp amount 0..5 (default 0.6), 0 = off\n"
        "  --no-sharpen           same as --sharpen 0\n"
        "  --radius R             unsharp sigma in output pixels 0.3..5 (default 1.0)\n"
        "  --threshold T          unsharp threshold 0..255 (default 0)\n"
        "  --premul               input is premultiplied alpha (clamp colour <= alpha)\n"
        "  --bench N              warm up once, then time N runs and report ms\n"
        "  --eval                 PSNR of box-downscale -> upscale vs. original\n",
        argv0);
}

static int parse_cli(int argc, char **argv, Cli *c)
{
    memset(c, 0, sizeof *c);
    upscaler_options_default(&c->opt);
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strncmp(a, "--", 2) != 0) {
            if (positional == 0) c->in = a;
            else if (positional == 1) c->out = a;
            else return -1;
            ++positional;
            continue;
        }
#define NEED_VALUE() do { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a); return -1; } } while (0)
        if (!strcmp(a, "--scale"))          { NEED_VALUE(); c->opt.scale = atoi(argv[++i]); }
        else if (!strcmp(a, "--kernel"))    {
            NEED_VALUE(); ++i;
            if (!strcmp(argv[i], "catmull") || !strcmp(argv[i], "bicubic")) c->opt.kernel = UPSCALE_KERNEL_CATMULL_ROM;
            else if (!strcmp(argv[i], "lanczos")) c->opt.kernel = UPSCALE_KERNEL_LANCZOS3;
            else { fprintf(stderr, "unknown kernel '%s'\n", argv[i]); return -1; }
        }
        else if (!strcmp(a, "--sharpen"))   { NEED_VALUE(); c->opt.sharpen_amount = (float)atof(argv[++i]); }
        else if (!strcmp(a, "--no-sharpen")) { c->opt.sharpen_amount = 0.0f; }
        else if (!strcmp(a, "--radius"))    { NEED_VALUE(); c->opt.sharpen_radius = (float)atof(argv[++i]); }
        else if (!strcmp(a, "--threshold")) { NEED_VALUE(); c->opt.sharpen_threshold = atoi(argv[++i]); }
        else if (!strcmp(a, "--premul"))    { c->opt.flags |= UPSCALE_FLAG_PREMULTIPLIED_ALPHA; }
        else if (!strcmp(a, "--bench"))     { NEED_VALUE(); c->bench = atoi(argv[++i]); }
        else if (!strcmp(a, "--eval"))      { c->eval = 1; }
        else { fprintf(stderr, "unknown option %s\n", a); return -1; }
#undef NEED_VALUE
    }
    if (c->in == NULL) return -1;
    if (c->out == NULL && !c->eval && c->bench == 0) return -1;
    return 0;
}

/* ---- synthetic test pattern (sharp rings, checker, lines, gentle noise) ---- */
static void gen_pattern(uint8_t *p, int w, int h)
{
    uint32_t s = 0x1234567u;
    const double cx = w * 0.5, cy = h * 0.5;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const double fx = (double)x / (w > 1 ? w - 1 : 1);
            const double fy = (double)y / (h > 1 ? h - 1 : 1);
            double r = 40.0 + 180.0 * fx, g = 40.0 + 180.0 * fy, b = 200.0 - 120.0 * fx * fy;
            const double dist = sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            if (fmod(dist, 48.0) < 2.5) r = g = b = 245.0;                      /* thin bright rings */
            if (x < w / 4 && y < h / 4) r = g = b = (((x / 8) + (y / 8)) & 1) ? 235.0 : 20.0; /* checker */
            if (((x + h - y) % 64) < 2) r = g = b = 15.0;                       /* diagonal lines */
            s = s * 1664525u + 1013904223u;
            const double n = (double)((s >> 24) & 15) / 2.0 - 4.0;             /* +-4 grain */
            const double v[3] = { r + n, g + n, b + n };
            for (int c = 0; c < 3; ++c)
                p[((size_t)y * (size_t)w + (size_t)x) * 3u + (size_t)c] =
                    (uint8_t)(v[c] < 0.0 ? 0.0 : (v[c] > 255.0 ? 255.0 : v[c]));
        }
    }
}

/* Loaded input. `pixels` is released with stbi_image_free (from_stb) or free. */
typedef struct Input {
    uint8_t *pixels;
    int      w, h, ch;
    int      from_stb;
} Input;

static int load_input(const char *spec, Input *in)
{
    memset(in, 0, sizeof *in);
    if (strncmp(spec, "synthetic:", 10) == 0) {
        int w = 0, h = 0;
        if (sscanf(spec + 10, "%dx%d", &w, &h) != 2 || w < 1 || h < 1 || w > 16384 || h > 16384) {
            fprintf(stderr, "bad synthetic spec, expected synthetic:WxH\n");
            return -1;
        }
        in->pixels = (uint8_t *)malloc((size_t)w * (size_t)h * 3u);
        if (!in->pixels) { fprintf(stderr, "out of memory\n"); return -1; }
        gen_pattern(in->pixels, w, h);
        in->w = w; in->h = h; in->ch = 3;
        return 0;
    }
    int n = 0;
    in->pixels = stbi_load(spec, &in->w, &in->h, &n, 0);
    if (in->pixels && n == 2) {                       /* gray+alpha: reload as RGBA to keep it simple */
        stbi_image_free(in->pixels);
        in->pixels = stbi_load(spec, &in->w, &in->h, &n, 4);
    }
    if (!in->pixels) { fprintf(stderr, "cannot load '%s': %s\n", spec, stbi_failure_reason()); return -1; }
    in->ch = n; in->from_stb = 1;
    return 0;
}

static void free_input(Input *in)
{
    if (in->pixels) { if (in->from_stb) stbi_image_free(in->pixels); else free(in->pixels); }
    memset(in, 0, sizeof *in);
}

static int has_ext(const char *path, const char *ext)
{
    const size_t lp = strlen(path), le = strlen(ext);
    if (lp < le) return 0;
    for (size_t i = 0; i < le; ++i) {
        char a = path[lp - le + i], b = ext[i];
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (a != b) return 0;
    }
    return 1;
}

static int write_image(const char *path, const ImageBuffer *im)
{
    if (has_ext(path, ".png"))
        return stbi_write_png(path, im->width, im->height, im->channels, im->data, im->stride) ? 0 : -1;

    /* jpg / bmp writers want tightly packed rows. */
    const size_t row = (size_t)im->width * (size_t)im->channels;
    uint8_t *tight = (uint8_t *)malloc(row * (size_t)im->height);
    if (!tight) return -1;
    for (int y = 0; y < im->height; ++y) memcpy(tight + row * (size_t)y, im->data + (size_t)y * (size_t)im->stride, row);
    int ok = 0;
    if (has_ext(path, ".jpg") || has_ext(path, ".jpeg"))
        ok = stbi_write_jpg(path, im->width, im->height, im->channels, tight, 95);
    else if (has_ext(path, ".bmp"))
        ok = stbi_write_bmp(path, im->width, im->height, im->channels, tight);
    else
        fprintf(stderr, "unsupported output extension (use .png/.jpg/.bmp)\n");
    free(tight);
    return ok ? 0 : -1;
}

/* ---- benchmark ------------------------------------------------------------ */
static int run_bench(const ImageBuffer *src, const UpscaleOptions *o, const ImageBuffer *dst, int iters)
{
    double *tot = (double *)malloc(sizeof(double) * (size_t)iters);
    double *res = (double *)malloc(sizeof(double) * (size_t)iters);
    double *pst = (double *)malloc(sizeof(double) * (size_t)iters);
    if (!tot || !res || !pst) { free(tot); free(res); free(pst); return UPSCALE_ERR_OOM; }

    int32_t rc = upscale_image_into(src, o, dst);                    /* warm-up: page-in, caches, branch predictors */
    for (int i = 0; i < iters && rc == UPSCALE_OK; ++i) {
        const double t0 = now_ms();
        rc = upscale_resample_rows(src, o->scale, o->kernel, dst, 0, dst->height);
        const double t1 = now_ms();
        if (rc == UPSCALE_OK) rc = upscale_postprocess(dst, o);
        const double t2 = now_ms();
        res[i] = t1 - t0; pst[i] = t2 - t1; tot[i] = t2 - t0;
    }
    if (rc == UPSCALE_OK) {
        double sum = 0.0;
        for (int i = 0; i < iters; ++i) sum += tot[i];
        qsort(tot, (size_t)iters, sizeof(double), cmp_double);
        qsort(res, (size_t)iters, sizeof(double), cmp_double);
        qsort(pst, (size_t)iters, sizeof(double), cmp_double);
        const double mp = (double)dst->width * (double)dst->height / 1e6;
        printf("benchmark  %dx%d -> %dx%d  (%d ch, %s, sharpen %.2f)  %d runs\n",
               src->width, src->height, dst->width, dst->height, src->channels,
               o->kernel == UPSCALE_KERNEL_LANCZOS3 ? "lanczos3" : "catmull-rom",
               (double)o->sharpen_amount, iters);
        printf("  total      min %8.2f ms   median %8.2f ms   mean %8.2f ms\n", tot[0], tot[iters / 2], sum / iters);
        printf("  resample   min %8.2f ms   median %8.2f ms\n", res[0], res[iters / 2]);
        printf("  sharpen    min %8.2f ms   median %8.2f ms\n", pst[0], pst[iters / 2]);
        printf("  throughput %.1f Mpixel/s (output, best run)\n", mp / (tot[0] / 1e3));
    }
    free(tot); free(res); free(pst);
    return rc;
}

/* ---- quality evaluation: PSNR against ground truth ------------------------- */
static double psnr(const ImageBuffer *a, const ImageBuffer *b)
{
    const int nc = (a->channels >= 3) ? 3 : 1;                       /* colour channels only */
    double se = 0.0;
    for (int y = 0; y < a->height; ++y) {
        const uint8_t *pa = a->data + (size_t)y * (size_t)a->stride;
        const uint8_t *pb = b->data + (size_t)y * (size_t)b->stride;
        for (int x = 0; x < a->width; ++x)
            for (int c = 0; c < nc; ++c) {
                const double d = (double)pa[(size_t)x * (size_t)a->channels + (size_t)c] -
                                 (double)pb[(size_t)x * (size_t)b->channels + (size_t)c];
                se += d * d;
            }
    }
    const double mse = se / ((double)a->width * a->height * nc);
    return mse <= 0.0 ? 99.0 : 10.0 * log10(255.0 * 255.0 / mse);
}

static int run_eval(const Input *in, const UpscaleOptions *base)
{
    const int s = base->scale;
    const int w = in->w / s * s, h = in->h / s * s, ch = in->ch;
    if (w < 8 * s || h < 8 * s) { fprintf(stderr, "image too small for --eval\n"); return UPSCALE_ERR_INVALID_ARGS; }

    ImageBuffer hr = {0}, lr = {0}, out = {0};
    int32_t rc = create_image_buffer(w, h, ch, &hr);
    if (rc == UPSCALE_OK) rc = create_image_buffer(w / s, h / s, ch, &lr);
    if (rc == UPSCALE_OK) rc = create_image_buffer(w, h, ch, &out);
    if (rc != UPSCALE_OK) goto done;

    for (int y = 0; y < h; ++y)                                       /* crop to multiple of scale */
        memcpy(hr.data + (size_t)y * (size_t)hr.stride, in->pixels + (size_t)y * (size_t)in->w * (size_t)ch, (size_t)w * (size_t)ch);

    for (int y = 0; y < lr.height; ++y)                               /* box (area) downscale */
        for (int x = 0; x < lr.width; ++x)
            for (int c = 0; c < ch; ++c) {
                int sum = 0;
                for (int dy = 0; dy < s; ++dy)
                    for (int dx = 0; dx < s; ++dx)
                        sum += hr.data[(size_t)(y * s + dy) * (size_t)hr.stride + (size_t)(x * s + dx) * (size_t)ch + (size_t)c];
                lr.data[(size_t)y * (size_t)lr.stride + (size_t)x * (size_t)ch + (size_t)c] = (uint8_t)((sum + s * s / 2) / (s * s));
            }

    printf("PSNR vs. original (higher is better), %dx -> box-downscale -> upscale, %dx%d, %d ch\n", s, w, h, ch);
    {   /* nearest-neighbour baseline */
        const double t0 = now_ms();
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                memcpy(out.data + (size_t)y * (size_t)out.stride + (size_t)x * (size_t)ch,
                       lr.data + (size_t)(y / s) * (size_t)lr.stride + (size_t)(x / s) * (size_t)ch, (size_t)ch);
        printf("  %-34s %6.2f dB  %8.2f ms\n", "nearest neighbour (baseline)", psnr(&hr, &out), now_ms() - t0);
    }
    const struct { const char *name; int kernel; float amount; } cfg[] = {
        { "catmull-rom",                UPSCALE_KERNEL_CATMULL_ROM, 0.0f },
        { "lanczos3",                   UPSCALE_KERNEL_LANCZOS3,    0.0f },
        { "catmull-rom + unsharp",      UPSCALE_KERNEL_CATMULL_ROM, base->sharpen_amount },
        { "lanczos3 + unsharp",         UPSCALE_KERNEL_LANCZOS3,    base->sharpen_amount },
    };
    for (size_t i = 0; i < sizeof cfg / sizeof cfg[0]; ++i) {
        UpscaleOptions o = *base;
        o.kernel = cfg[i].kernel;
        o.sharpen_amount = cfg[i].amount;
        const double t0 = now_ms();
        rc = upscale_image_into(&lr, &o, &out);
        const double t1 = now_ms();
        if (rc != UPSCALE_OK) goto done;
        printf("  %-34s %6.2f dB  %8.2f ms\n", cfg[i].name, psnr(&hr, &out), t1 - t0);
    }
done:
    free_image_buffer(&hr); free_image_buffer(&lr); free_image_buffer(&out);
    return rc;
}

/* ---- main ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    Cli cli;
    if (parse_cli(argc, argv, &cli) != 0) { usage(argv[0]); return 2; }

    printf("upscaler engine v%d.%d.%d\n", UPSCALER_VERSION_MAJOR, UPSCALER_VERSION_MINOR, UPSCALER_VERSION_PATCH);

    Input in;
    if (load_input(cli.in, &in) != 0) return 1;

    int exit_code = 1;
    ImageBuffer dst = {0};
    int32_t rc = UPSCALE_OK;

    /* Non-owning view over the stb pixels: tight rows, stride = w*ch. */
    const ImageBuffer src = { in.pixels, in.w, in.h, in.ch, in.w * in.ch };

    if (cli.eval) {
        rc = run_eval(&in, &cli.opt);
        if (rc != UPSCALE_OK) fprintf(stderr, "eval failed: %s\n", upscaler_error_string(rc));
        exit_code = (rc == UPSCALE_OK) ? 0 : 1;
        goto cleanup;
    }

    rc = create_image_buffer(in.w * cli.opt.scale, in.h * cli.opt.scale, in.ch, &dst);
    if (rc != UPSCALE_OK) { fprintf(stderr, "cannot allocate output: %s\n", upscaler_error_string(rc)); goto cleanup; }

    if (cli.bench > 0) {
        rc = run_bench(&src, &cli.opt, &dst, cli.bench);
    } else {
        const double t0 = now_ms();
        rc = upscale_image_into(&src, &cli.opt, &dst);
        const double t1 = now_ms();
        if (rc == UPSCALE_OK)
            printf("upscaled %dx%d -> %dx%d (%d ch) in %.2f ms\n", in.w, in.h, dst.width, dst.height, in.ch, t1 - t0);
    }
    if (rc != UPSCALE_OK) { fprintf(stderr, "upscale failed: %s (%d)\n", upscaler_error_string(rc), rc); goto cleanup; }

    if (cli.out) {
        if (write_image(cli.out, &dst) != 0) { fprintf(stderr, "cannot write '%s'\n", cli.out); goto cleanup; }
        printf("wrote %s\n", cli.out);
    }
    exit_code = 0;

cleanup:
    free_image_buffer(&dst);
    free_input(&in);
    return exit_code;
}
