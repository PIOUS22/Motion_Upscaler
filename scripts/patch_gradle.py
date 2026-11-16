#!/usr/bin/env python3
"""Prepares a freshly generated Flutter android/app for the C upscaler:
  1. build.gradle(.kts): CMake hook, lint workaround, minSdk >= 24 (needed by the `gal` plugin)
  2. AndroidManifest.xml: legacy storage permission for saving images on Android 9 and below
Usage: patch_gradle.py <android/app dir>.   Idempotent; handles Kotlin DSL and Groovy DSL.
(A second defaultConfig{} block is legal in Gradle: it configures the same object again.)"""
import pathlib, re, sys

MIN_SDK = 24
app = pathlib.Path(sys.argv[1])
kts, groovy = app / "build.gradle.kts", app / "build.gradle"

KTS_BLOCK = '''
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
    // Release lint ("lintVital") sometimes crashes on fresh projects; it does not affect the APK.
    lint {
        checkReleaseBuilds = false
        abortOnError = false
    }
    // --- end ---
'''
GROOVY_BLOCK = '''
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
    lint {
        checkReleaseBuilds false
        abortOnError false
    }
    // --- end ---
'''

if kts.exists():
    path, is_kts, block = kts, True, KTS_BLOCK
elif groovy.exists():
    path, is_kts, block = groovy, False, GROOVY_BLOCK
else:
    sys.exit(f"no build.gradle(.kts) in {app}")

text = path.read_text()
if "native/CMakeLists.txt" not in text:
    text, n = re.subn(r"^android\s*\{[ \t]*\n", lambda m: m.group(0) + block, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"could not find 'android {{' in {path}")

# minSdk: raise to MIN_SDK whether the template uses flutter.minSdkVersion or a literal number.
if f"{MIN_SDK})" not in text.split("minSdk", 1)[-1][:80]:
    wrap = (lambda e: f"maxOf({e}, {MIN_SDK})") if is_kts else (lambda e: f"Math.max({e}, {MIN_SDK})")
    text, n1 = re.subn(r"(minSdk(?:Version)?\s*=?\s*)flutter\.minSdkVersion\b",
                       lambda m: m.group(1) + wrap("flutter.minSdkVersion"), text)
    n2 = 0
    if n1 == 0:
        text, n2 = re.subn(r"(minSdk(?:Version)?\s*=?\s*)(\d+)\b",
                           lambda m: m.group(1) + str(max(int(m.group(2)), MIN_SDK)), text)
    print(f"minSdk adjusted ({n1 + n2} occurrence(s))")
path.write_text(text)
print("patched", path)

manifest = app / "src" / "main" / "AndroidManifest.xml"
if manifest.exists():
    m = manifest.read_text()
    if "WRITE_EXTERNAL_STORAGE" not in m:
        perm = ('    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"\n'
                '        android:maxSdkVersion="29" />\n')
        m, n = re.subn(r"(\s*)<application", lambda mm: "\n" + perm + "    <application", m, count=1)
        if n == 1:
            manifest.write_text(m)
            print("patched", manifest)
        else:
            sys.exit("could not find <application> in manifest")
    else:
        print("manifest already patched")
else:
    print("warning: no AndroidManifest.xml at", manifest)
