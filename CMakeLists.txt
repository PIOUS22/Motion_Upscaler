# ==============================================================================
# upscaler_engine -- pure C11 image upscaler
#
#  Android (Gradle -> externalNativeBuild) : builds libupscaler.so only
#  Desktop Linux/macOS                     : libupscaler.so + test_engine + upscaler_cli
#
#  Desktop quick start:
#      scripts/fetch_stb.sh                       # one-off: downloads stb_image*.h
#      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
#      ctest --test-dir build --output-on-failure
# ==============================================================================
cmake_minimum_required(VERSION 3.18)
project(upscaler_engine VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)                 # -std=c11, not gnu11: keeps us honest
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(UPSCALER_BUILD_CLI      "Build the desktop CLI and tests (ignored on Android)" ON)
option(UPSCALER_NATIVE_TUNING  "Desktop only: compile with -march=native"            OFF)

set(UPSCALER_SOURCES src/upscaler_engine.c)

# ---- shared compile settings -------------------------------------------------
function(upscaler_configure_target tgt)
  target_include_directories(${tgt} PUBLIC include)

  target_compile_options(${tgt} PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes -Wvla
    -ffunction-sections -fdata-sections
    # -O3 turns on the loop auto-vectoriser (GCC) and keeps it on (Clang). Debug stays -O0 -g
    # so Valgrind / debuggers see real variables.
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Debug>:-g>
    $<$<NOT:$<CONFIG:Debug>>:-O3>)

  # ---- SIMD ------------------------------------------------------------------
  # arm64-v8a : Advanced SIMD (NEON) is part of the base AArch64 ISA. There is nothing to
  #             enable; -O3 makes the compiler emit NEON for the u8 x s16 -> s32 MAC loops.
  # armeabi-v7a: NEON is optional in the ABI, so request it explicitly.
  if(ANDROID_ABI STREQUAL "armeabi-v7a" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7")
    target_compile_options(${tgt} PRIVATE -mfpu=neon)
    set(UPSCALER_SIMD "NEON (armv7, -mfpu=neon)" PARENT_SCOPE)
  elseif(ANDROID_ABI STREQUAL "arm64-v8a" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(UPSCALER_SIMD "NEON (AArch64 baseline)" PARENT_SCOPE)
  else()
    set(UPSCALER_SIMD "host default (auto-vectorised)" PARENT_SCOPE)
    if(UPSCALER_NATIVE_TUNING AND NOT ANDROID)
      target_compile_options(${tgt} PRIVATE -march=native)
    endif()
  endif()
endfunction()

# ---- the shared library (what Flutter / JNI loads) ---------------------------
add_library(upscaler SHARED ${UPSCALER_SOURCES})
target_compile_definitions(upscaler PRIVATE UPSCALER_BUILD_SHARED)
set_target_properties(upscaler PROPERTIES
  C_VISIBILITY_PRESET hidden          # only UPSCALER_API symbols are exported
  VISIBILITY_INLINES_HIDDEN ON)
upscaler_configure_target(upscaler)
message(STATUS "upscaler: SIMD path = ${UPSCALER_SIMD}")

if(NOT WIN32)
  target_link_libraries(upscaler PRIVATE m)         # libm only; libc is implicit
endif()

if(ANDROID)
  target_link_options(upscaler PRIVATE
    -Wl,--gc-sections
    -Wl,--no-undefined                 # proves: zero unresolved dependencies
    -Wl,-z,max-page-size=16384)        # 16 KB page-size devices (required for Android 15+ on Play)
elseif(UNIX AND NOT APPLE)
  target_link_options(upscaler PRIVATE -Wl,--gc-sections -Wl,--no-undefined)
endif()

# ---- desktop-only: tests + CLI -----------------------------------------------
if(NOT ANDROID AND UPSCALER_BUILD_CLI)
  add_library(upscaler_static STATIC ${UPSCALER_SOURCES})
  upscaler_configure_target(upscaler_static)
  target_link_libraries(upscaler_static PUBLIC m)

  add_executable(test_engine tests/test_engine.c)
  target_link_libraries(test_engine PRIVATE upscaler_static)
  enable_testing()
  add_test(NAME engine_selftest COMMAND test_engine)

  if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/stb_image.h" AND
     EXISTS "${CMAKE_SOURCE_DIR}/third_party/stb_image_write.h")
    add_executable(upscaler_cli tools/main.c tools/stb_impl.c)
    target_include_directories(upscaler_cli PRIVATE third_party)
    target_link_libraries(upscaler_cli PRIVATE upscaler_static)
    # Strict warnings for our code only, not for the stb implementation file.
    set_source_files_properties(tools/main.c PROPERTIES
      COMPILE_OPTIONS "-Wall;-Wextra;-Wpedantic;-Wshadow;-Wstrict-prototypes")
    target_compile_options(upscaler_cli PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O3>)
    add_test(NAME cli_smoke COMMAND upscaler_cli synthetic:200x150
             ${CMAKE_BINARY_DIR}/smoke.png --scale 4 --kernel lanczos --sharpen 0.6)
    add_test(NAME cli_eval  COMMAND upscaler_cli synthetic:256x192 --eval --scale 2)
  else()
    message(WARNING "third_party/stb_image*.h missing -> upscaler_cli not built. Run scripts/fetch_stb.sh")
  endif()
endif()
