// Minimal test screen: pick a photo, upscale 2x with the C engine, show it and the timing.
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'upscaler_service.dart';

void main() => runApp(const MaterialApp(home: Home()));

class Home extends StatefulWidget {
  const Home({super.key});
  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  ui.Image? _out;
  String _info = 'Tap the button and pick a photo (1-2 MP is a good start)';
  bool _busy = false;
  int _scale = 2;

  Future<void> _run() async {
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
      input.dispose();
      _out?.dispose();
      setState(() {
        _out = out;
        _info = '${out.width ~/ _scale}x${out.height ~/ _scale} -> ${out.width}x${out.height} in $ms ms';
      });
    } on UpscalerException catch (e) {
      setState(() => _info = 'Engine error: $e');
    } catch (e) {
      setState(() => _info = 'Error: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('C Upscaler')),
        body: Column(children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: Row(children: [
              Expanded(child: Text(_info)),
              SegmentedButton<int>(
                segments: const [ButtonSegment(value: 2, label: Text('2x')), ButtonSegment(value: 4, label: Text('4x'))],
                selected: {_scale},
                onSelectionChanged: (s) => setState(() => _scale = s.first),
              ),
            ]),
          ),
          if (_busy) const LinearProgressIndicator(),
          Expanded(
            child: _out == null ? const SizedBox() : InteractiveViewer(maxScale: 8, child: RawImage(image: _out)),
          ),
        ]),
        floatingActionButton: FloatingActionButton(
          onPressed: _busy ? null : _run,
          child: const Icon(Icons.add_photo_alternate),
        ),
      );
}
