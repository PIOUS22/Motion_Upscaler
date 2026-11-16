// upscaler_service.dart -- Flutter-facing API on top of the raw bindings.
//
// DATA FLOW (and where bytes are actually copied)
//
//   ui.Image --toByteData(rawRgba)--> ByteData        copy #1  (Flutter's API; unavoidable)
//   ByteData --row copy-->  NATIVE src buffer          copy #2  (into 64-byte aligned C memory)
//   C engine: src -> dst (native, never touches the Dart heap)   0 copies, runs in worker isolate(s)
//   dst.data.asTypedList() -> decodeImageFromPixels    copy #3  (Flutter's API; unavoidable)
//
// Between #2 and #3 only POINTER VALUES (ints) cross isolate boundaries, never pixel data.
// If you can obtain pixels from a native source instead of ui.Image (camera plane, Android
// Bitmap, a file decoded by native code), decode directly into `src.data` and copy #1/#2
// disappear -- see NativeImage.view.
import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:isolate';
import 'dart:ui' as ui;

import 'package:ffi/ffi.dart' show Utf8Pointer, calloc;

import 'upscaler_ffi.dart';

class UpscalerException implements Exception {
  UpscalerException(this.code, this.message);
  final int code;
  final String message;
  @override
  String toString() => 'UpscalerException($code): $message';
}

/// Native image owned by this object. ALWAYS call [dispose] (use try/finally).
/// Dart GC does not know about the native bytes, so there is deliberately no finalizer
/// pretending to manage them.
class NativeImage {
  NativeImage._(this._b, this.ptr);

  final UpscalerBindings _b;
  final ffi.Pointer<ImageBuffer> ptr; // calloc'd struct; its ->data is engine-owned
  bool _disposed = false;

  factory NativeImage.create(UpscalerBindings b, int w, int h, {int channels = 4}) {
    final p = calloc<ImageBuffer>();
    final rc = b.createImageBuffer(w, h, channels, p);
    if (rc != kUpscaleOk) {
      calloc.free(p);
      throw UpscalerException(rc, b.errorString(rc).toDartString());
    }
    return NativeImage._(b, p);
  }

  int get width => ptr.ref.width;
  int get height => ptr.ref.height;
  int get stride => ptr.ref.stride;

  /// Zero-copy window onto the native pixels (stride*height bytes). Valid until [dispose].
  ffi.Pointer<ffi.Uint8> get data => ptr.ref.data;
  // ignore: avoid_returning_this
  List<int> get view => data.asTypedList(stride * height);

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _b.freeImageBuffer(ptr); // releases pixels, zeroes struct
    calloc.free(ptr); // releases the struct itself
  }
}

class UpscaleSettings {
  const UpscaleSettings({
    this.scale = 2,
    this.kernel = kKernelLanczos3,
    this.sharpenAmount = 0.6,
    this.sharpenRadius = 1.0,
    this.sharpenThreshold = 0,
    this.workers = 1,
    this.maxOutputPixels = 64 * 1000 * 1000,
  });
  final int scale;
  final int kernel;
  final double sharpenAmount;
  final double sharpenRadius;
  final int sharpenThreshold;

  /// Isolates used for the resample pass (the sharpen pass is single-threaded by design).
  final int workers;

  /// Guard. The C engine allows 1 GiB per buffer, which is far more than a phone can spare:
  /// 64 MP RGBA = 256 MB for the output alone. Larger inputs need tiling (not implemented).
  final int maxOutputPixels;
}

class Upscaler {
  Upscaler() : _b = UpscalerBindings.open();
  final UpscalerBindings _b;

  /// Upscales a ui.Image. Returned image is owned by the caller (call .dispose()).
  Future<ui.Image> upscale(ui.Image input, [UpscaleSettings s = const UpscaleSettings()]) async {
    final w = input.width, h = input.height;
    if (w * s.scale * h * s.scale > s.maxOutputPixels) {
      throw UpscalerException(-4, 'output exceeds maxOutputPixels; tile the image first');
    }

    // rawRgba == PREMULTIPLIED alpha (Flutter docs) -> tell the engine so ringing can't
    // push colour above alpha.
    final bd = await input.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (bd == null) throw UpscalerException(-1, 'toByteData returned null');

    NativeImage? src, dst;
    final opts = calloc<UpscaleOptions>();
    try {
      src = NativeImage.create(_b, w, h);
      dst = NativeImage.create(_b, w * s.scale, h * s.scale);

      // copy #2: honour the native stride (rows are padded to 16 bytes)
      final srcBytes = src.data.asTypedList(src.stride * h);
      final rowBytes = w * 4;
      for (var y = 0; y < h; y++) {
        srcBytes.setRange(y * src.stride, y * src.stride + rowBytes,
            bd.buffer.asUint8List(bd.offsetInBytes + y * rowBytes, rowBytes));
      }

      _b.optionsDefault(opts);
      opts.ref
        ..scale = s.scale
        ..kernel = s.kernel
        ..flags = kFlagPremultipliedAlpha
        ..sharpenAmount = s.sharpenAmount
        ..sharpenRadius = s.sharpenRadius
        ..sharpenThreshold = s.sharpenThreshold;

      // Only integers cross the isolate boundary.
      final srcAddr = src.ptr.address, dstAddr = dst.ptr.address, optAddr = opts.address;
      final dstRows = h * s.scale, n = s.workers.clamp(1, 8);

      if (n == 1) {
        await Isolate.run(() => _check(UpscalerBindings.open(), (b) => b.upscaleImageInto(
            ffi.Pointer.fromAddress(srcAddr),
            ffi.Pointer.fromAddress(optAddr),
            ffi.Pointer.fromAddress(dstAddr))));
      } else {
        // Bands write disjoint rows of the SAME dst: safe because the engine is stateless and
        // bit-identical to a single pass (verified by T3 in tests/test_engine.c).
        await Future.wait([
          for (var i = 0; i < n; i++)
            Isolate.run(() {
              final r0 = dstRows * i ~/ n, r1 = dstRows * (i + 1) ~/ n;
              return _check(UpscalerBindings.open(), (b) => b.resampleRows(
                  ffi.Pointer.fromAddress(srcAddr), s.scale, s.kernel,
                  ffi.Pointer.fromAddress(dstAddr), r0, r1));
            }),
        ]);
        await Isolate.run(() => _check(UpscalerBindings.open(), (b) => b.postprocess(
            ffi.Pointer.fromAddress(dstAddr), ffi.Pointer.fromAddress(optAddr))));
      }

      // copy #3 happens inside decodeImageFromPixels (it copies into an ImmutableBuffer),
      // so freeing `dst` right after the callback fires is safe.
      final done = Completer<ui.Image>();
      ui.decodeImageFromPixels(dst.data.asTypedList(dst.stride * dst.height), dst.width,
          dst.height, ui.PixelFormat.rgba8888, done.complete,
          rowBytes: dst.stride);
      return await done.future;
    } finally {
      src?.dispose();
      dst?.dispose();
      calloc.free(opts);
    }
  }

  static void _check(UpscalerBindings b, int Function(UpscalerBindings) call) {
    final rc = call(b);
    if (rc != kUpscaleOk) {
      throw UpscalerException(rc, b.errorString(rc).toDartString());
    }
  }
}
