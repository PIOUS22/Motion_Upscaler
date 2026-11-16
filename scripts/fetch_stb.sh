#!/usr/bin/env bash
# Downloads the two single-header stb libraries used ONLY by the desktop CLI
# (tools/main.c). The engine itself and the Android .so never include them.
set -euo pipefail
cd "$(dirname "$0")/../third_party"
for f in stb_image.h stb_image_write.h; do
  curl -fsSL -o "$f" "https://raw.githubusercontent.com/nothings/stb/master/$f"
  echo "fetched $f ($(wc -l < "$f") lines)"
done
