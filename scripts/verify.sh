#!/usr/bin/env bash
# One-shot verification on Ubuntu: strict C11 build, self-tests, ASan+UBSan,
# Valgrind (memcheck, full leak check), benchmark, quality (PSNR), OOM behaviour.
#   scripts/verify.sh            # everything
#   BENCH_ITERS=30 scripts/verify.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/verify
mkdir -p "$OUT"
WARN="-Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes -Wvla"
STD="-std=c11 -Iinclude"
BENCH_ITERS="${BENCH_ITERS:-15}"

[ -f third_party/stb_image.h ] || bash scripts/fetch_stb.sh

stb_obj() {  # $1 = extra flags ; compiled with warnings off (third-party code)
  gcc -std=c11 -w -Ithird_party $1 -c tools/stb_impl.c -o "$OUT/stb_impl_$2.o"
}

echo "== 1/6  gcc -O3, strict C11, zero warnings allowed"
gcc $STD $WARN -Werror -O3 src/upscaler_engine.c tests/test_engine.c -o "$OUT/test_engine" -lm
stb_obj "-O2" rel
gcc $STD $WARN -Werror -O3 -Ithird_party src/upscaler_engine.c tools/main.c "$OUT/stb_impl_rel.o" -o "$OUT/upscaler_cli" -lm
"$OUT/test_engine" | tail -3
"$OUT/upscaler_cli" synthetic:640x360 "$OUT/synthetic_x2.png" --scale 2 >/dev/null && echo "CLI smoke ok -> $OUT/synthetic_x2.png"

echo; echo "== 2/6  AddressSanitizer + UndefinedBehaviorSanitizer (halt on first error)"
SAN="-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer"
gcc $STD $WARN -O1 -g $SAN src/upscaler_engine.c tests/test_engine.c -o "$OUT/test_engine_asan" -lm
"$OUT/test_engine_asan" | tail -2
stb_obj "-O1 $SAN" asan
gcc $STD -O1 -g $SAN -Ithird_party src/upscaler_engine.c tools/main.c "$OUT/stb_impl_asan.o" -o "$OUT/cli_asan" -lm
"$OUT/cli_asan" synthetic:97x61 "$OUT/asan.png" --scale 4 --kernel catmull >/dev/null && echo "ASan/UBSan CLI run clean"

echo; echo "== 3/6  Valgrind memcheck (leaks + invalid reads/writes + uninitialised use)"
gcc $STD $WARN -O1 -g src/upscaler_engine.c tests/test_engine.c -o "$OUT/test_engine_vg" -lm
stb_obj "-O1 -g" vg
gcc $STD -O1 -g -Ithird_party src/upscaler_engine.c tools/main.c "$OUT/stb_impl_vg.o" -o "$OUT/cli_vg" -lm
VG="valgrind --error-exitcode=99 --leak-check=full --show-leak-kinds=all --errors-for-leak-kinds=all --track-origins=yes -q"
$VG "$OUT/test_engine_vg" | tail -2
$VG "$OUT/cli_vg" synthetic:64x48 "$OUT/vg.png" --scale 4 --sharpen 1.0 >/dev/null && echo "Valgrind CLI run clean (0 errors, 0 leaks)"

echo; echo "== 4/6  Benchmark (release build)"
"$OUT/upscaler_cli" synthetic:1920x1080 --scale 2 --bench "$BENCH_ITERS"
"$OUT/upscaler_cli" synthetic:1920x1080 --scale 2 --kernel catmull --bench "$BENCH_ITERS"
"$OUT/upscaler_cli" synthetic:1000x1000 --scale 4 --bench "$BENCH_ITERS"

echo; echo "== 5/6  Quality: PSNR vs. ground truth"
"$OUT/upscaler_cli" synthetic:1024x768 --eval --scale 2
"$OUT/upscaler_cli" synthetic:1024x768 --eval --scale 4

echo; echo "== 6/6  Out-of-memory path (address space capped at 60 MB; output alone needs 108 MB)"
set +e
msg=$( ( ulimit -v 60000; "$OUT/upscaler_cli" synthetic:1500x1500 "$OUT/never.png" --scale 4 ) 2>&1 )
rc=$?
set -e
echo "$msg" | tail -1
# must be a clean exit(1) carrying the engine's OOM message -- not a signal (rc>=128) and not a success
if [ "$rc" -eq 1 ] && echo "$msg" | grep -q "out of memory"; then
  echo "UPSCALE_ERR_OOM surfaced as a clean error (rc=1), no crash, no output written"
else
  echo "OOM test FAILED (rc=$rc)"; exit 1
fi

echo; echo "ALL CHECKS PASSED"
