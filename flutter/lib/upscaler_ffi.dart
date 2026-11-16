// upscaler_ffi.dart -- raw dart:ffi bindings. 1:1 with include/upscaler_engine.h.
// Keep this file dumb: no logic, only layout + symbol lookup.
//
// Struct layouts (LP64 / arm64-v8a) are pinned by _Static_asserts in the C engine:
//   ImageBuffer    : data@0 (8) width@8 height@12 channels@16 stride@20   => 24 bytes
//   UpscaleOptions : scale@0 kernel@4 flags@8 amount@12 radius@16 thr@20  => 24 bytes
// Both are only used on 64-bit ABIs here (arm64-v8a, x86_64). Pointer-size-dependent
// layout is handled by dart:ffi itself, so armeabi-v7a would also be correct.
import 'dart:ffi' as ffi;
import 'dart:io' show Platform;

import 'package:ffi/ffi.dart' show Utf8;

const int kUpscaleOk = 0;
const int kKernelCatmullRom = 0;
const int kKernelLanczos3 = 1;
const int kFlagPremultipliedAlpha = 1;

final class ImageBuffer extends ffi.Struct {
  external ffi.Pointer<ffi.Uint8> data;
  @ffi.Int32()
  external int width;
  @ffi.Int32()
  external int height;
  @ffi.Int32()
  external int channels;
  @ffi.Int32()
  external int stride;
}

final class UpscaleOptions extends ffi.Struct {
  @ffi.Int32()
  external int scale;
  @ffi.Int32()
  external int kernel;
  @ffi.Int32()
  external int flags;
  @ffi.Float()
  external double sharpenAmount;
  @ffi.Float()
  external double sharpenRadius;
  @ffi.Int32()
  external int sharpenThreshold;
}

typedef _CreateC = ffi.Int32 Function(ffi.Int32, ffi.Int32, ffi.Int32, ffi.Pointer<ImageBuffer>);
typedef _CreateD = int Function(int, int, int, ffi.Pointer<ImageBuffer>);
typedef _FreeC = ffi.Void Function(ffi.Pointer<ImageBuffer>);
typedef _FreeD = void Function(ffi.Pointer<ImageBuffer>);
typedef _IntoC = ffi.Int32 Function(
    ffi.Pointer<ImageBuffer>, ffi.Pointer<UpscaleOptions>, ffi.Pointer<ImageBuffer>);
typedef _IntoD = int Function(
    ffi.Pointer<ImageBuffer>, ffi.Pointer<UpscaleOptions>, ffi.Pointer<ImageBuffer>);
typedef _RowsC = ffi.Int32 Function(ffi.Pointer<ImageBuffer>, ffi.Int32, ffi.Int32,
    ffi.Pointer<ImageBuffer>, ffi.Int32, ffi.Int32);
typedef _RowsD = int Function(
    ffi.Pointer<ImageBuffer>, int, int, ffi.Pointer<ImageBuffer>, int, int);
typedef _PostC = ffi.Int32 Function(ffi.Pointer<ImageBuffer>, ffi.Pointer<UpscaleOptions>);
typedef _PostD = int Function(ffi.Pointer<ImageBuffer>, ffi.Pointer<UpscaleOptions>);
typedef _DefaultsC = ffi.Void Function(ffi.Pointer<UpscaleOptions>);
typedef _DefaultsD = void Function(ffi.Pointer<UpscaleOptions>);
typedef _ErrStrC = ffi.Pointer<Utf8> Function(ffi.Int32);
typedef _ErrStrD = ffi.Pointer<Utf8> Function(int);

/// Symbol table. Cheap to construct; every isolate builds its own (DynamicLibrary handles
/// are per-isolate, the underlying .so is mapped once per process).
class UpscalerBindings {
  UpscalerBindings._(ffi.DynamicLibrary lib)
      : createImageBuffer = lib.lookupFunction<_CreateC, _CreateD>('create_image_buffer'),
        freeImageBuffer = lib.lookupFunction<_FreeC, _FreeD>('free_image_buffer'),
        upscaleImageInto = lib.lookupFunction<_IntoC, _IntoD>('upscale_image_into'),
        resampleRows = lib.lookupFunction<_RowsC, _RowsD>('upscale_resample_rows'),
        postprocess = lib.lookupFunction<_PostC, _PostD>('upscale_postprocess'),
        optionsDefault = lib.lookupFunction<_DefaultsC, _DefaultsD>('upscaler_options_default'),
        errorString = lib.lookupFunction<_ErrStrC, _ErrStrD>('upscaler_error_string');

  factory UpscalerBindings.open() {
    final ffi.DynamicLibrary lib = Platform.isAndroid || Platform.isLinux
        ? ffi.DynamicLibrary.open('libupscaler.so')
        : Platform.isIOS || Platform.isMacOS
            ? ffi.DynamicLibrary.process() // statically linked into the app binary
            : throw UnsupportedError('unsupported platform');
    return UpscalerBindings._(lib);
  }

  final _CreateD createImageBuffer;
  final _FreeD freeImageBuffer;
  final _IntoD upscaleImageInto;
  final _RowsD resampleRows;
  final _PostD postprocess;
  final _DefaultsD optionsDefault;
  final _ErrStrD errorString;
}
