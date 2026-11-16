# upscaler_engine — pure C11 offline image upscaler (NDK + Dart FFI)

```
upscaler/
├── include/upscaler_engine.h     public ABI (fixed-width ints, plain structs, int32 status codes)
├── src/upscaler_engine.c         the whole engine: libc + libm only
├── CMakeLists.txt                Android (.so only) and desktop (.so + tests + CLI)
├── tools/main.c, stb_impl.c      CLI: load/save via stb, --bench, --eval (PSNR)
├── tests/test_engine.c           9,356 checks vs. an independent double-precision reference
├── tests/test_abi.py             ctypes: struct sizes, exported symbols, DT_NEEDED
├── scripts/verify.sh             gcc strict -> ASan/UBSan -> Valgrind -> bench -> PSNR -> OOM
├── scripts/fetch_stb.sh          downloads stb_image*.h (CLI only, never in the .so)
└── flutter/                      upscaler_ffi.dart, upscaler_service.dart, android_build.gradle.snippet
```

## Run it (Ubuntu)
```
scripts/fetch_stb.sh                     # once (stb headers already included in third_party/)
scripts/verify.sh                        # everything; non-zero exit on any failure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build
build/upscaler_cli photo.jpg out.png --scale 2 --kernel lanczos --sharpen 0.6 --radius 1.0
build/upscaler_cli photo.jpg --scale 2 --bench 20
```

## Architecture

**Algorithm.** Output pixel `x = q·s + r` samples source position `q + (2r+1−s)/(2s)`. The fractional part depends
only on `r`, so a 2× upscale needs **2 filters** and 4× needs **4** (polyphase). Catmull-Rom (4 taps) or Lanczos-3
(6 taps) weights are computed once per call, quantised to Q14 `int16` with the sum forced to exactly 16384
(flat areas stay bit-exact), then the hot loops are branch-free `u8 × s16 → s32` multiply-accumulates.

**Memory math (1920×1080 RGBA, 2×).**
| | naive separable | this engine |
|---|---|---|
| intermediate image | 3840·1080·4 = **16.6 MB** (blows every cache) | 8-row ring: 8·15,360 B = **123 KB** |
| padded source row | – | (1920+6)·4 ≈ 7.7 KB |
| total scratch | 16.6 MB | **≈ 131 KB** (fits L2), allocated once, freed on the single exit path |

Source rows are read once, sequentially. Edge handling: each source row is copied into a scratch row padded with
`taps/2` replicated edge pixels, and vertical indices are clamped before ring lookup ⇒ no clamping in the inner
loops and no out-of-bounds read is possible.

**Buffers.** `create_image_buffer`: zero-filled, 64-byte-aligned data, `stride = align16(w·ch)`, hidden back-pointer
so `free_image_buffer` needs no extra ABI field; frees are idempotent. Sizes are validated with 64-bit math
(`UPSCALE_ERR_TOO_LARGE` beyond 32768 px/side or 1 GiB). A struct pointing at foreign memory is a *view* and must
never be passed to `free_image_buffer`.

**Unsharp mask.** Separable Gaussian (symmetric kernel ⇒ mirrored taps are added before multiplying), streamed with
the same ring technique, in place, alpha untouched, optional threshold, branch-free combine.

**Threading.** No globals/statics. `upscale_resample_rows(src, …, dst, row_begin, row_end)` can run on N threads or
Dart isolates against the same `dst` and is bit-identical to one full pass (test T3); then call
`upscale_postprocess` once.

## Flutter boundary — where bytes are copied
`ui.Image → toByteData` (copy, Flutter API) → row-copy into native `src` (copy) → **C engine: src→dst, no copies,
off the UI isolate; only pointer integers cross isolates** → `decodeImageFromPixels(rowBytes: stride)` (copy, Flutter
API). The engine boundary itself is zero-copy; if your pixels come from native code (camera, native decoder),
decode straight into `NativeImage.data` and the first two copies vanish. JNI is deliberately not provided:
`AndroidBitmap_lockPixels` needs `libjnigraphics`, which breaks the zero-dependency rule — if you want it, keep
it in a separate shim library that calls this one.

## What was verified here, and what was not
Verified in this sandbox (x86-64, GCC 13): strict `-std=c11 -Wall -Wextra -Wpedantic -Werror` clean; 9,356 checks
pass (engine vs. independent reference ≤ 2 levels for Lanczos / ≤ 1 for Catmull-Rom, which is the rounding budget of
two 8-bit stages; flat images exact; banding exact; every error code; padding never written); ASan+UBSan clean;
Valgrind memcheck clean with `--leak-check=full` (0 errors, 0 leaks); real OOM path returns `UPSCALE_ERR_OOM`;
built `.so` exports only the 10 API symbols and depends only on libc/libm.

Measured on x86-64 (not a phone): 1920×1080 → 3840×2160 RGB, Lanczos-3 + sharpen ≈ 141 ms (59 MP/s out);
resample alone ≈ 50 ms, sharpen ≈ 90 ms. The sharpen pass dominates; for speed, use radius ≤ 1 or `--no-sharpen`.

**Not verified:** (1) NEON was checked only by compiling to AArch64/ARMv7 assembly with Clang (widening
`smlal`/`vmlal` and 128-bit loads present; ARMv7 has none without `-mfpu=neon`) — I could not run it, so **time it on
your device**. (2) The Android Gradle build was not run (no NDK here). (3) The Dart files were not compiled (no Dart
SDK here) — they follow dart:ffi's documented API, but expect to fix small typos. The ctypes test proves the
symbol names and struct layouts they rely on. (4) Alpha: `rawRgba` is documented as premultiplied and the code
assumes `decodeImageFromPixels` reads `rgba8888` the same way — check a semi-transparent PNG once.

## Limits and honest caveats
* No tiling: output is capped at 64 MP in Dart (`maxOutputPixels`); a 12 MP photo at 4× is 768 MB and will not fit
  on most phones. Add tile-with-overlap if you need that (overlap ≥ 3 px for resample + 15 px for sharpen).
* This is classical interpolation: it sharpens edges but cannot invent detail like a neural upscaler. The `--eval`
  PSNR on the synthetic pattern (noisy by design) is only a smoke test; judge quality on real photos.
