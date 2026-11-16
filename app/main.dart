// Test screen: pick a photo -> upscale with the C engine -> compare before/after -> save to gallery.
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:gal/gal.dart';
import 'package:image_picker/image_picker.dart';

import 'upscaler_service.dart';

void main() => runApp(const MaterialApp(home: Home()));

enum ViewMode { after, before, split }

class Home extends StatefulWidget {
  const Home({super.key});
  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  ui.Image? _orig;
  ui.Image? _out;
  String _info = 'Tap the photo button (bottom right) and pick an image';
  bool _busy = false;
  int _scale = 2;
  ViewMode _mode = ViewMode.after;
  double _split = 0.5;
  // Shared by After/Before so the zoom and position stay the same when you switch.
  final TransformationController _tc = TransformationController();

  @override
  void dispose() {
    _tc.dispose();
    _orig?.dispose();
    _out?.dispose();
    super.dispose();
  }

  void _toast(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(msg)));
  }

  Future<void> _pickAndUpscale() async {
    final f = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (f == null) return;
    setState(() {
      _busy = true;
      _info = 'Working...';
    });
    try {
      final codec = await ui.instantiateImageCodec(await f.readAsBytes());
      final input = (await codec.getNextFrame()).image;
      final sw = Stopwatch()..start();
      final out = await Upscaler().upscale(input, UpscaleSettings(scale: _scale, workers: 4));
      final ms = sw.elapsedMilliseconds;
      final oldO = _orig, oldU = _out;
      setState(() {
        _orig = input;
        _out = out;
        _mode = ViewMode.after;
        _tc.value = Matrix4.identity();
        _info = 'Original ${input.width}x${input.height}  ->  Result ${out.width}x${out.height}  (${_scale}x, $ms ms)';
      });
      oldO?.dispose();
      oldU?.dispose();
    } on UpscalerException catch (e) {
      setState(() => _info = 'Engine error: $e');
    } catch (e) {
      setState(() => _info = 'Error: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _save() async {
    final out = _out;
    if (out == null) return;
    setState(() => _busy = true);
    try {
      final bd = await out.toByteData(format: ui.ImageByteFormat.png);
      if (bd == null) throw 'PNG encoding failed';
      await Gal.putImageBytes(
        bd.buffer.asUint8List(bd.offsetInBytes, bd.lengthInBytes),
        name: 'upscaled_${DateTime.now().millisecondsSinceEpoch}',
      );
      _toast('Saved to your gallery (Pictures)');
    } catch (e) {
      _toast('Save failed: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Widget _imageFill(ui.Image i) =>
      SizedBox.expand(child: RawImage(image: i, fit: BoxFit.contain, filterQuality: FilterQuality.medium));

  Widget _tag(String text) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(12)),
        child: Text(text, style: const TextStyle(color: Colors.white, fontSize: 12)),
      );

  Widget _viewer() {
    final out = _out, orig = _orig;
    if (out == null || orig == null) return const SizedBox();

    if (_mode == ViewMode.split) {
      return LayoutBuilder(builder: (context, box) {
        final w = box.maxWidth;
        return GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTapDown: (d) => setState(() => _split = (d.localPosition.dx / w).clamp(0.0, 1.0)),
          onHorizontalDragUpdate: (d) =>
              setState(() => _split = (_split + d.delta.dx / w).clamp(0.0, 1.0)),
          child: Stack(children: [
            Positioned.fill(child: _imageFill(out)),
            Positioned.fill(child: ClipRect(clipper: _LeftClip(_split), child: _imageFill(orig))),
            Positioned(left: w * _split - 1, top: 0, bottom: 0, child: Container(width: 2, color: Colors.white)),
            Positioned(left: w * _split - 14, top: 0, bottom: 0, child: const Center(child: Icon(Icons.swap_horiz, color: Colors.white))),
            Positioned(left: 8, top: 8, child: _tag('Before')),
            Positioned(right: 8, top: 8, child: _tag('After')),
          ]),
        );
      });
    }

    final showAfter = _mode == ViewMode.after;
    return Stack(children: [
      Positioned.fill(
        child: InteractiveViewer(
          transformationController: _tc,
          maxScale: 16,
          child: _imageFill(showAfter ? out : orig),
        ),
      ),
      Positioned(left: 8, top: 8, child: _tag(showAfter ? 'After (upscaled)' : 'Before (original)')),
    ]);
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(
          title: const Text('C Upscaler'),
          actions: [
            IconButton(
              tooltip: 'Save to gallery',
              icon: const Icon(Icons.download),
              onPressed: (_out == null || _busy) ? null : _save,
            ),
          ],
        ),
        body: Column(children: [
          Padding(padding: const EdgeInsets.fromLTRB(12, 8, 12, 4), child: Text(_info)),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Wrap(spacing: 12, runSpacing: 6, children: [
              SegmentedButton<int>(
                segments: const [
                  ButtonSegment(value: 2, label: Text('2x')),
                  ButtonSegment(value: 4, label: Text('4x')),
                ],
                selected: {_scale},
                onSelectionChanged: _busy ? null : (s) => setState(() => _scale = s.first),
              ),
              SegmentedButton<ViewMode>(
                segments: const [
                  ButtonSegment(value: ViewMode.after, label: Text('After')),
                  ButtonSegment(value: ViewMode.before, label: Text('Before')),
                  ButtonSegment(value: ViewMode.split, label: Text('Split')),
                ],
                selected: {_mode},
                onSelectionChanged: _out == null ? null : (s) => setState(() => _mode = s.first),
              ),
            ]),
          ),
          if (_out != null)
            const Padding(
              padding: EdgeInsets.fromLTRB(12, 4, 12, 0),
              child: Text(
                'Tip: pinch to zoom in, then tap After / Before to compare the same spot. '
                'Split: drag the line.',
                style: TextStyle(fontSize: 12, color: Colors.black54),
              ),
            ),
          if (_busy) const LinearProgressIndicator(),
          const SizedBox(height: 4),
          Expanded(child: ClipRect(child: _viewer())),
        ]),
        floatingActionButton: FloatingActionButton(
          onPressed: _busy ? null : _pickAndUpscale,
          child: const Icon(Icons.add_photo_alternate),
        ),
      );
}

class _LeftClip extends CustomClipper<Rect> {
  _LeftClip(this.fraction);
  final double fraction;
  @override
  Rect getClip(Size size) => Rect.fromLTRB(0, 0, size.width * fraction, size.height);
  @override
  bool shouldReclip(_LeftClip old) => old.fraction != fraction;
}
