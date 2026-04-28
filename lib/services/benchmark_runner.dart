import 'dart:io' if (dart.library.html) 'io_stub.dart';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart' show Color, Offset, Paint, Rect;
import 'package:battery_plus/battery_plus.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:thermal/thermal.dart';
import 'inference_service.dart';
import 'picsprompt_generation_service.dart';

/// What to run, repeated [iterations] times per (model × EP).
class BenchmarkSpec {
  final String modelId; // 'animegan', 'butterflies', 'lcm'
  final List<ExecutionProvider> providers;
  final int iterations;
  final int warmup;

  /// Cooldown between iterations. Without this, mid-range SoCs (MediaTek
  /// Dimensity, Snapdragon 6-series) thermally throttle within a few runs and
  /// the NNAPI driver can OOM/crash. 8s is the sweet spot for AnimeGAN; bump to
  /// 20s+ for diffusion models.
  final Duration cooldownPerIter;

  /// Longer pause when switching EPs — gives the NNAPI compiler time to release
  /// device handles and the SoC time to drop a thermal step.
  final Duration cooldownPerEp;

  const BenchmarkSpec({
    required this.modelId,
    this.providers = const [
      ExecutionProvider.cpu,
      ExecutionProvider.xnnpack,
      ExecutionProvider.nnapi,
    ],
    this.iterations = 5,
    this.warmup = 1,
    this.cooldownPerIter = const Duration(seconds: 8),
    this.cooldownPerEp = const Duration(seconds: 20),
  });
}

class BenchmarkRow {
  final String runId;
  final String device;
  final String modelId;
  final String executionProvider;
  final int iteration;
  final int totalMs;
  final int inferenceMs;
  final int ramMB;
  final int resolution;
  final int steps;
  final double batteryPctBefore;
  final double batteryPctAfter;
  final ThermalStatus thermalBefore;
  final ThermalStatus thermalAfter;
  final String tsIso;
  final bool ok;
  final String? error;

  BenchmarkRow({
    required this.runId,
    required this.device,
    required this.modelId,
    required this.executionProvider,
    required this.iteration,
    required this.totalMs,
    required this.inferenceMs,
    required this.ramMB,
    required this.resolution,
    required this.steps,
    required this.batteryPctBefore,
    required this.batteryPctAfter,
    required this.thermalBefore,
    required this.thermalAfter,
    required this.tsIso,
    required this.ok,
    this.error,
  });

  static const List<String> csvHeader = [
    'run_id', 'device', 'model', 'ep', 'iter',
    'total_ms', 'infer_ms', 'ram_mb', 'resolution', 'steps',
    'battery_pct_before', 'battery_pct_after',
    'thermal_before', 'thermal_after', 'ts', 'ok', 'error',
  ];

  String toCsvRow() {
    String q(Object? v) {
      final s = v == null ? '' : v.toString();
      return '"${s.replaceAll('"', '""')}"';
    }
    return [
      q(runId), q(device), q(modelId), q(executionProvider), iteration,
      totalMs, inferenceMs, ramMB, resolution, steps,
      batteryPctBefore, batteryPctAfter,
      q(thermalBefore.name), q(thermalAfter.name),
      q(tsIso), ok, q(error ?? ''),
    ].join(',');
  }
}

/// Drives `InferenceService.generate*` over each (model × EP × iter), capturing
/// app-level timing + battery + thermal. The host-side `bench_android.py` script
/// pulls system metrics in parallel via adb and merges by `run_id`/timestamp.
///
/// Output: one JSONL line per iteration appended to
///   `<appDocs>/benchmark_<runId>.jsonl`
/// Returned [String] is the absolute path to that file so the UI can share it.
class BenchmarkRunner {
  final InferenceService _infer = InferenceService();
  final Battery _battery = Battery();
  final Thermal _thermal = Thermal();

  /// Cached 512×512 PNG used as the AnimeGAN test input. Built once via
  /// `dart:ui` so we know it round-trips through Skia (the hand-rolled byte
  /// blob we shipped first had a busted IDAT CRC).
  Uint8List? _animeganTestPng;

  Future<String> run({
    required List<BenchmarkSpec> specs,
    String? prompt,
  }) async {
    final runId = DateTime.now().millisecondsSinceEpoch.toString();
    final docs = await getApplicationDocumentsDirectory();
    final outFile = File('${docs.path}/benchmark_$runId.csv');
    final sink = outFile.openWrite(mode: FileMode.write);
    sink.writeln(BenchmarkRow.csvHeader.join(','));

    final deviceName = await _readDeviceName();

    try {
      for (final spec in specs) {
        for (int epIdx = 0; epIdx < spec.providers.length; epIdx++) {
          final ep = spec.providers[epIdx];
          if (epIdx > 0) {
            await Future<void>.delayed(spec.cooldownPerEp);
          }
          // Warmup runs are timed but flagged so the analysis can skip them.
          for (int i = -spec.warmup; i < spec.iterations; i++) {
            if (i > -spec.warmup) {
              await Future<void>.delayed(spec.cooldownPerIter);
            }
            final batBefore = await _safeBattery();
            final thBefore = await _safeThermal();
            final ts = DateTime.now().toIso8601String();

            GenerationResult? r;
            String? err;
            try {
              r = await _runOnce(spec.modelId, ep, prompt: prompt);
            } catch (e) {
              err = e.toString();
            }

            final batAfter = await _safeBattery();
            final thAfter = await _safeThermal();

            final row = BenchmarkRow(
              runId: runId,
              device: deviceName,
              modelId: spec.modelId,
              executionProvider: ep.wire,
              iteration: i, // negative = warmup
              totalMs: r?.totalTimeMs ?? 0,
              inferenceMs: r?.inferenceTimeMs ?? 0,
              ramMB: r?.ramUsedMB ?? 0,
              resolution: r?.resolution ?? 0,
              steps: r?.inferenceSteps ?? 0,
              batteryPctBefore: batBefore,
              batteryPctAfter: batAfter,
              thermalBefore: thBefore,
              thermalAfter: thAfter,
              tsIso: ts,
              ok: err == null,
              error: err,
            );
            sink.writeln(row.toCsvRow());
            await sink.flush();
          }
        }
      }
    } finally {
      await sink.close();
    }
    return outFile.path;
  }

  Future<GenerationResult> _runOnce(
    String modelId,
    ExecutionProvider ep, {
    String? prompt,
  }) async {
    switch (modelId) {
      case 'animegan':
        // Uses a built-in 512×512 test image so the bench does not depend on a
        // gallery pick. Image content does not affect AnimeGAN compute cost —
        // it is a fixed-size single-pass GAN — so a simple gradient is fine.
        _animeganTestPng ??= await _buildTestPng(512, 512);
        return _infer.generateAnimeganAvatar(
          inputImageBytes: _animeganTestPng!,
          executionProvider: ep,
        );
      case 'butterflies':
        return _infer.generate(
          prompt: '',
          modelId: 'butterflies',
          steps: 50,
          executionProvider: ep,
        );
      case 'lcm':
        return _infer.generate(
          prompt: prompt ?? 'a scenic mountain landscape at sunset',
          modelId: 'lcm',
          resolution: 256,
          steps: 4,
          executionProvider: ep,
        );
      default:
        throw ArgumentError('Unknown modelId: $modelId');
    }
  }

  Future<double> _safeBattery() async {
    try {
      return (await _battery.batteryLevel).toDouble();
    } catch (_) {
      return -1;
    }
  }

  Future<ThermalStatus> _safeThermal() async {
    try {
      return await _thermal.thermalStatus;
    } catch (_) {
      return ThermalStatus.none;
    }
  }

  Future<String> _readDeviceName() async {
    try {
      final info = DeviceInfoPlugin();
      final a = await info.androidInfo;
      return '${a.manufacturer} ${a.model} (SDK ${a.version.sdkInt})';
    } catch (_) {
      return 'unknown';
    }
  }

  /// Build a real PNG via Skia so AnimeGAN's BitmapFactory.decodeByteArray
  /// always succeeds. Two diagonal colour bands keep the centre-crop interesting
  /// without affecting compute cost.
  static Future<Uint8List> _buildTestPng(int w, int h) async {
    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(
      recorder,
      Rect.fromLTWH(0, 0, w.toDouble(), h.toDouble()),
    );
    canvas.drawRect(
      Rect.fromLTWH(0, 0, w.toDouble(), h.toDouble()),
      Paint()..color = const Color(0xFF606080),
    );
    canvas.drawCircle(
      Offset(w / 2, h / 2),
      w * 0.35,
      Paint()..color = const Color(0xFFE0C090),
    );
    final picture = recorder.endRecording();
    final image = await picture.toImage(w, h);
    final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    image.dispose();
    picture.dispose();
    if (byteData == null) {
      throw StateError('Could not encode benchmark PNG');
    }
    return byteData.buffer.asUint8List();
  }
}
