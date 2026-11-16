#!/usr/bin/env python3
"""Adds the CMake/NDK hook to a freshly generated Flutter android/app/build.gradle(.kts).
Usage: patch_gradle.py <android/app dir>
Idempotent. Works for both the Kotlin DSL (.kts) and the Groovy DSL. A second defaultConfig{}
block is legal in Gradle: it configures the same object again."""
import pathlib, re, sys

app = pathlib.Path(sys.argv[1])
kts, groovy = app / "build.gradle.kts", app / "build.gradle"
if kts.exists():
    path, text = kts, kts.read_text()
    block = '''
    // --- added for the C upscaler engine ---
    defaultConfig {
        externalNativeBuild {
            cmake { arguments += listOf("-DANDROID_STL=none", "-DUPSCALER_BUILD_CLI=OFF") }
        }
    }
    externalNativeBuild {
        cmake {
            path = file("../../native/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    // --- end ---
'''
elif groovy.exists():
    path, text = groovy, groovy.read_text()
    block = '''
    // --- added for the C upscaler engine ---
    defaultConfig {
        externalNativeBuild {
            cmake { arguments "-DANDROID_STL=none", "-DUPSCALER_BUILD_CLI=OFF" }
        }
    }
    externalNativeBuild {
        cmake {
            path "../../native/CMakeLists.txt"
            version "3.22.1"
        }
    }
    // --- end ---
'''
else:
    sys.exit(f"no build.gradle(.kts) in {app}")

if "native/CMakeLists.txt" in text:
    print("already patched:", path)
    sys.exit(0)
new, n = re.subn(r"^android\s*\{[ \t]*\n", lambda m: m.group(0) + block, text, count=1, flags=re.M)
if n != 1:
    sys.exit(f"could not find 'android {{' in {path}")
path.write_text(new)
print("patched", path)
