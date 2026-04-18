import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:io' if (dart.library.html) 'io_stub.dart';
import 'package:path_provider/path_provider.dart';
import 'picsprompt_config.dart';
import 'picsprompt_generation_service.dart';

/// One generation's worth of bytes + metrics. The numeric fields feed into
/// TestLogger so the same struct backs both the chat UI and the paper CSV.
class GenerationResult {
  GenerationResult({
    required this.imageBytes,
    required this.modelUsed,
    required this.totalTimeMs,
    required this.modelLoadTimeMs,
    required this.inferenceTimeMs,
    required this.ramUsedMB,
    required this.resolution,
    required this.inferenceSteps,
    this.executionProvider = ExecutionProvider.cpu,
  });

  final Uint8List imageBytes;
  final String modelUsed;
  final int totalTimeMs;
  final int modelLoadTimeMs;
  final int inferenceTimeMs;
  final int ramUsedMB;
  final int resolution;
  final int inferenceSteps;
  final ExecutionProvider executionProvider;
}

class InferenceService {
  static final InferenceService _instance = InferenceService._internal();
  factory InferenceService() => _instance;
  InferenceService._internal();

  /// Subfolders under `models/dreamshaper/` (same order as runtime resolution).
  static const List<String> dreamshaperVariantDirs = [
    'onnx_quint8',
    'onnx_int8_mobile',
    'onnx_int8',
    'onnx',
    'onnx_wqdq',
  ];

  /// Butterflies has only one variant for now; resolution check still requires
  /// `unet/model.onnx` to be present (and externally-stored weights file).
  static const List<String> butterfliesVariantDirs = ['onnx'];

  /// AnimeGAN ONNX file under `models/animegan/`. Prefers INT8 for faster
  /// inference on device; falls back to FP32 if INT8 is not present.
  static const List<String> animeganOnnxFiles = [
    'face_paint_512_v2_int8.onnx',
    'face_paint_512_v2_fp32.onnx',
  ];

  PicspromptGenerationService? _service;
  String? _activeModelDir;
  String? _activeModelId;

  // Lazily initialise the native service (loads config once). Re-inits the
  // native bundle when the resolved directory changes (e.g. an incomplete
  // `onnx_wqdq` is skipped in favour of `onnx_quint8`, or model switches
  // between dreamshaper and butterflies).
  Future<PicspromptGenerationService> _getService({
    required String modelId,
    required String modelDir,
  }) async {
    final config = await PicspromptConfigLoader().load();
    _service ??= PicspromptGenerationService(config);
    if (_activeModelDir != modelDir || _activeModelId != modelId) {
      await _service!.initModel(modelDir: modelDir);
      _activeModelDir = modelDir;
      _activeModelId = modelId;
    }
    return _service!;
  }

  // Bundle layout matches `picsprompt-models/models/dreamshaper/<variant>/`.
  // Try ~1GB QUInt8 first, then Android-safe `onnx_int8_mobile` (~3GB), then
  // QInt8 / FP32. `onnx_int8` is listed after `onnx_int8_mobile` so a device
  // with both does not prefer the QInt8 graph that often fails on Android
  // (ConvInteger).
  // Override for local testing:
  //   `flutter run --dart-define=PICSPROMPT_MODEL_DIR=/path/to/bundle`.
  static Future<String?> dreamshaperModelDirOrNull() async {
    if (kIsWeb) return null;
    const override = String.fromEnvironment('PICSPROMPT_MODEL_DIR');
    if (override.isNotEmpty) {
      return _isDreamshaperBundleComplete(override) ? override : null;
    }
    final String baseDir;
    if (Platform.isAndroid) {
      final extDir = await getExternalStorageDirectory();
      if (extDir == null) return null;
      baseDir = extDir.path;
    } else {
      baseDir = (await getApplicationDocumentsDirectory()).path;
    }
    for (final v in dreamshaperVariantDirs) {
      final dirPath = '$baseDir/models/dreamshaper/$v';
      if (_isDreamshaperBundleComplete(dirPath)) return dirPath;
    }
    return null;
  }

  static Future<bool> hasCompleteDreamshaperBundle() async =>
      await dreamshaperModelDirOrNull() != null;

  /// Butterflies bundle resolver. Layout:
  /// `[docs|extStorage]/models/butterflies/[variant]/unet/model.onnx` (+ .data)
  static Future<String?> butterfliesModelDirOrNull() async {
    if (kIsWeb) return null;
    const override = String.fromEnvironment('PICSPROMPT_BUTTERFLIES_DIR');
    if (override.isNotEmpty) {
      return _isButterfliesBundleComplete(override) ? override : null;
    }
    final String baseDir;
    if (Platform.isAndroid) {
      final extDir = await getExternalStorageDirectory();
      if (extDir == null) return null;
      baseDir = extDir.path;
    } else {
      baseDir = (await getApplicationDocumentsDirectory()).path;
    }
    for (final v in butterfliesVariantDirs) {
      final dirPath = '$baseDir/models/butterflies/$v';
      if (_isButterfliesBundleComplete(dirPath)) return dirPath;
    }
    return null;
  }

  static Future<bool> hasCompleteButterfliesBundle() async =>
      await butterfliesModelDirOrNull() != null;

  /// Resolves a usable AnimeGAN ONNX file (int8 preferred, fp32 fallback).
  /// Layout: `[docs|extStorage]/models/animegan/face_paint_512_v2_*.onnx`.
  static Future<String?> animeganModelPathOrNull() async {
    if (kIsWeb) return null;
    const override = String.fromEnvironment('PICSPROMPT_ANIMEGAN_PATH');
    if (override.isNotEmpty) {
      return File(override).existsSync() ? override : null;
    }
    final String baseDir;
    if (Platform.isAndroid) {
      final extDir = await getExternalStorageDirectory();
      if (extDir == null) return null;
      baseDir = extDir.path;
    } else {
      baseDir = (await getApplicationDocumentsDirectory()).path;
    }
    for (final name in animeganOnnxFiles) {
      final path = '$baseDir/models/animegan/$name';
      // Tiny model (~2 MB int8 / ~9 MB fp32); 1 MB floor catches half-pushed files.
      if (File(path).existsSync() && File(path).lengthSync() > 1 * 1024 * 1024) {
        return path;
      }
    }
    return null;
  }

  static Future<bool> hasAnimeganModel() async =>
      await animeganModelPathOrNull() != null;

  /// External weights next to `*.onnx`: ORT may emit `model.onnx_data` or `model.onnx.data`.
  static bool _onnxReadyWithOptionalExternal(File onnxFile,
      {int stubMaxBytes = 8 * 1024 * 1024}) {
    if (!onnxFile.existsSync()) return false;
    if (onnxFile.lengthSync() >= stubMaxBytes) return true;
    final d = onnxFile.parent.path;
    final n = onnxFile.uri.pathSegments.last;
    return File('$d/${n}_data').existsSync() ||
        File('$d/$n.data').existsSync();
  }

  /// Butterflies bundle is just the UNet (no tokenizer / no VAE).
  /// `unet/model.onnx` may be a small ONNX stub when external weights are used,
  /// so we accept either a >8MB self-contained graph or a `.data` sidecar.
  static bool _isButterfliesBundleComplete(String root) {
    final unet = File('$root/unet/model.onnx');
    return _onnxReadyWithOptionalExternal(unet);
  }

  /// True only if [root] has tokenizer + the three ONNX sessions the native
  /// bridge loads. Skips empty or half-pushed folders.
  static bool _isDreamshaperBundleComplete(String root) {
    bool f(String rel) => File('$root/$rel').existsSync();
    if (!f('tokenizer/vocab.json') ||
        !f('tokenizer/merges.txt') ||
        !f('text_encoder/model.onnx') ||
        !f('unet/model.onnx') ||
        !f('vae_decoder/model.onnx')) {
      return false;
    }
    if (!_onnxReadyWithOptionalExternal(File('$root/text_encoder/model.onnx')) ||
        !_onnxReadyWithOptionalExternal(File('$root/unet/model.onnx')) ||
        !_onnxReadyWithOptionalExternal(File('$root/vae_decoder/model.onnx'))) {
      return false;
    }
    return true;
  }

  static String _basename(String path) {
    final n = path.replaceAll('\\', '/');
    final i = n.lastIndexOf('/');
    return i >= 0 ? n.substring(i + 1) : n;
  }

  Future<GenerationResult> generate({
    required String prompt,
    String modelId = 'lcm',
    int resolution = 256,
    /// LCM schedule cap is 50; DDPM caps at 1000 (native side enforces).
    int steps = 14,
    ExecutionProvider executionProvider = ExecutionProvider.cpu,
  }) async {
    final totalTimer = Stopwatch()..start();

    if (modelId == 'butterflies') {
      if (kIsWeb || !await hasCompleteButterfliesBundle()) {
        return _generationResultPlaceholder(
          modelId: 'butterflies',
          resolution: 128,
          steps: steps,
          totalTimer: totalTimer,
          executionProvider: executionProvider,
        );
      }
      return _generateWithButterflies(
        steps: steps,
        totalTimer: totalTimer,
        executionProvider: executionProvider,
      );
    }

    if (kIsWeb || !await hasCompleteDreamshaperBundle()) {
      return _generationResultPlaceholder(
        modelId: 'lcm',
        resolution: resolution,
        steps: steps,
        totalTimer: totalTimer,
        executionProvider: executionProvider,
      );
    }
    return _generateWithDreamshaper(
      prompt: prompt,
      resolution: resolution,
      steps: steps,
      totalTimer: totalTimer,
      executionProvider: executionProvider,
    );
  }

  Future<GenerationResult> _generateWithButterflies({
    required int steps,
    required Stopwatch totalTimer,
    ExecutionProvider executionProvider = ExecutionProvider.cpu,
  }) async {
    final modelDir = (await butterfliesModelDirOrNull())!;
    final service = await _getService(modelId: 'butterflies', modelDir: modelDir);

    final docsDir = await getApplicationDocumentsDirectory();
    final outputPath =
        '${docsDir.path}/butterflies_${DateTime.now().millisecondsSinceEpoch}.png';

    // 50 DDPM steps is the standard demo budget — high enough for crisp
    // butterflies, low enough to finish in a few seconds at 128² on mobile.
    final nativeSteps = steps.clamp(10, 1000);
    final request = GenerationRequest(
      prompt: '',
      negativePrompt: '',
      width: 128,
      height: 128,
      steps: nativeSteps,
      guidance: 0.0,
      seed: _dreamshaperSeed(),
      outputPath: outputPath,
      pipelineKind: 'ddpm_unconditional',
      executionProvider: executionProvider,
    );

    final inferenceTimer = Stopwatch()..start();
    final resultPath = await service.generate(request);
    inferenceTimer.stop();

    final imageBytes = await File(resultPath).readAsBytes();
    totalTimer.stop();
    return GenerationResult(
      imageBytes: imageBytes,
      modelUsed: 'butterflies:${_basename(modelDir)}',
      executionProvider: executionProvider,
      totalTimeMs: totalTimer.elapsedMilliseconds,
      modelLoadTimeMs: 0,
      inferenceTimeMs: inferenceTimer.elapsedMilliseconds,
      ramUsedMB: _readRAMUsage(),
      resolution: 128,
      inferenceSteps: nativeSteps,
    );
  }

  Future<GenerationResult> _generateWithDreamshaper({
    required String prompt,
    required int resolution,
    required int steps,
    required Stopwatch totalTimer,
    ExecutionProvider executionProvider = ExecutionProvider.cpu,
  }) async {
    final modelDir = (await dreamshaperModelDirOrNull())!;
    final service = await _getService(modelId: 'lcm', modelDir: modelDir);

    // Pick profile from requested resolution / step budget (chat default: 14 steps).
    final profileId = resolution >= 384
        ? 'quality'
        : steps >= 8
            ? 'balanced'
            : 'low';
    final profile = service.config.profilesById[profileId]!;

    const stepCap = 50;
    final nativeSteps = math.min(stepCap, math.max(profile.steps, steps));

    final docsDir = await getApplicationDocumentsDirectory();
    final outputPath =
        '${docsDir.path}/generated_${DateTime.now().millisecondsSinceEpoch}.png';

    // Avoid "face" / portrait-heavy negatives: in CLIP+CFG they still activate
    // face concepts and DreamShaper-LCM then drifts to anime portraits.
    const lcmNegative =
        'blurry, low quality, worst quality, jpeg artifacts, bad anatomy, '
        'deformed, disfigured, watermark, text, cropped, duplicate';

    final request = GenerationRequest(
      prompt: _dreamshaperPositivePrompt(prompt),
      negativePrompt: lcmNegative,
      width: profile.width,
      height: profile.height,
      steps: nativeSteps,
      guidance: profile.guidance,
      seed: _dreamshaperSeed(),
      outputPath: outputPath,
      pipelineKind: 'lcm',
      executionProvider: executionProvider,
    );

    final modelLoadTimer = Stopwatch()..start();
    // initModel is idempotent; native side skips reload if already loaded.
    modelLoadTimer.stop();

    final inferenceTimer = Stopwatch()..start();
    final resultPath = await service.generate(request);
    inferenceTimer.stop();

    final imageBytes = await File(resultPath).readAsBytes();
    totalTimer.stop();

    final bundleTag = 'lcm:${_basename(modelDir)}';

    return GenerationResult(
      imageBytes: imageBytes,
      modelUsed: bundleTag,
      totalTimeMs: totalTimer.elapsedMilliseconds,
      modelLoadTimeMs: modelLoadTimer.elapsedMilliseconds,
      inferenceTimeMs: inferenceTimer.elapsedMilliseconds,
      ramUsedMB: _readRAMUsage(),
      resolution: profile.width,
      inferenceSteps: nativeSteps,
      executionProvider: executionProvider,
    );
  }

  /// Img2img: stylize [inputImageBytes] with AnimeGANv2 face_paint_512_v2.
  /// Returns a placeholder result when the model is missing, so the UI can
  /// surface the same "push the bundle" message it shows for the other models.
  Future<GenerationResult> generateAnimeganAvatar({
    required Uint8List inputImageBytes,
    ExecutionProvider executionProvider = ExecutionProvider.cpu,
  }) async {
    final totalTimer = Stopwatch()..start();
    if (kIsWeb || !await hasAnimeganModel()) {
      return _generationResultPlaceholder(
        modelId: 'animegan',
        resolution: 512,
        steps: 1,
        totalTimer: totalTimer,
        executionProvider: executionProvider,
      );
    }
    final modelPath = (await animeganModelPathOrNull())!;
    final config = await PicspromptConfigLoader().load();
    _service ??= PicspromptGenerationService(config);

    final docsDir = await getApplicationDocumentsDirectory();
    final outputPath =
        '${docsDir.path}/avatar_${DateTime.now().millisecondsSinceEpoch}.png';

    final inferenceTimer = Stopwatch()..start();
    final resultPath = await _service!.generateAnimeGan(
      modelPath: modelPath,
      imageBytes: inputImageBytes,
      outputPath: outputPath,
      executionProvider: executionProvider,
    );
    inferenceTimer.stop();

    final imageBytes = await File(resultPath).readAsBytes();
    totalTimer.stop();

    return GenerationResult(
      imageBytes: imageBytes,
      modelUsed: 'animegan:${_basename(modelPath)}',
      totalTimeMs: totalTimer.elapsedMilliseconds,
      modelLoadTimeMs: 0,
      inferenceTimeMs: inferenceTimer.elapsedMilliseconds,
      ramUsedMB: _readRAMUsage(),
      resolution: 512,
      // AnimeGAN is a single-pass GAN, not a diffusion schedule.
      inferenceSteps: 1,
      executionProvider: executionProvider,
    );
  }

  /// Img2img using DreamShaper-LCM (the bundle already on device). Produces
  /// far better avatars than AnimeGAN: real face geometry, proper lighting,
  /// prompt-driven style. Strength 0.55 keeps identity; raise toward 0.7 for
  /// stronger stylization.
  Future<GenerationResult> generateLcmAvatar({
    required Uint8List inputImageBytes,
    String stylePrompt = '',
    // Lower strength keeps the subject (face, gender, pose) from the photo.
    // DreamShaper's portrait prior is strong enough that >0.5 starts swapping
    // the person out for its "default girl portrait" mean.
    double strength = 0.40,
    ExecutionProvider executionProvider = ExecutionProvider.cpu,
  }) async {
    final totalTimer = Stopwatch()..start();
    if (kIsWeb || !await hasCompleteDreamshaperBundle()) {
      return _generationResultPlaceholder(
        modelId: 'lcm_avatar',
        resolution: 512,
        steps: 8,
        totalTimer: totalTimer,
        executionProvider: executionProvider,
      );
    }
    final modelDir = (await dreamshaperModelDirOrNull())!;
    final service = await _getService(modelId: 'lcm', modelDir: modelDir);

    final docsDir = await getApplicationDocumentsDirectory();
    final outputPath =
        '${docsDir.path}/avatar_${DateTime.now().millisecondsSinceEpoch}.png';

    // Style-only prompt: in img2img the subject comes from the photo, so words
    // like "portrait of …" or "head-and-shoulders" pull DreamShaper toward its
    // generic-girl-portrait prior and the input person gets lost. Describe
    // *how* it should look, not *who* should be in it.
    const styleDefault =
        'stylized digital illustration, sharp focus, detailed, '
        'cinematic lighting, vibrant colors, clean lines';
    final composedPrompt = stylePrompt.trim().isEmpty
        ? styleDefault
        : '${stylePrompt.trim()}, $styleDefault';
    // Identity-preserving negatives: explicitly push away from the "default
    // anime girl" failure mode plus the usual quality issues.
    const negative =
        'different person, face swap, gender swap, anime girl, generic woman, '
        'blurry, lowres, deformed, extra limbs, bad anatomy, watermark, text, '
        'duplicate, jpeg artifacts, oversaturated';

    final inferenceTimer = Stopwatch()..start();
    final resultPath = await service.generateLcmImg2Img(
      imageBytes: inputImageBytes,
      prompt: composedPrompt,
      negativePrompt: negative,
      outputPath: outputPath,
      strength: strength,
      // More steps with low strength helps quality without losing identity —
      // each step covers a smaller fraction of the schedule.
      steps: 10,
      // 2.5 keeps prompt obedience without flipping LCM into mode-collapse.
      guidance: 2.5,
      width: 512,
      height: 512,
      executionProvider: executionProvider,
    );
    inferenceTimer.stop();

    final imageBytes = await File(resultPath).readAsBytes();
    totalTimer.stop();

    return GenerationResult(
      imageBytes: imageBytes,
      modelUsed: 'lcm_avatar:${_basename(modelDir)}',
      totalTimeMs: totalTimer.elapsedMilliseconds,
      modelLoadTimeMs: 0,
      inferenceTimeMs: inferenceTimer.elapsedMilliseconds,
      ramUsedMB: _readRAMUsage(),
      resolution: 512,
      inferenceSteps: 8,
      executionProvider: executionProvider,
    );
  }

  /// Returns a 1×1 PNG with a `:no-onnx-bundle` tag so the chat UI can show a
  /// helpful "push the model" message instead of a broken image.
  Future<GenerationResult> _generationResultPlaceholder({
    required String modelId,
    required int resolution,
    required int steps,
    required Stopwatch totalTimer,
    ExecutionProvider executionProvider = ExecutionProvider.cpu,
  }) async {
    totalTimer.stop();
    return GenerationResult(
      imageBytes: _placeholderImage(),
      modelUsed: '$modelId:no-onnx-bundle',
      totalTimeMs: totalTimer.elapsedMilliseconds,
      modelLoadTimeMs: 0,
      inferenceTimeMs: 0,
      ramUsedMB: _readRAMUsage(),
      resolution: resolution,
      inferenceSteps: steps,
      executionProvider: executionProvider,
    );
  }

  /// Nudges plain text2img away from default "portrait girl" collapse.
  String _dreamshaperPositivePrompt(String prompt) {
    if (prompt.toLowerCase().contains('coherent composition')) return prompt;
    return '$prompt, coherent composition, scene matches the description';
  }

  int _dreamshaperSeed() {
    final t = DateTime.now();
    final a = t.millisecondsSinceEpoch;
    final b = t.microsecondsSinceEpoch;
    return (a ^ b ^ (a << 13)) & 0x7FFFFFFF;
  }
}

int _readRAMUsage() {
  if (kIsWeb) return 0;
  try {
    final lines = File('/proc/self/status').readAsLinesSync();
    for (final line in lines) {
      if (line.startsWith('VmRSS')) {
        final kb =
            int.tryParse(line.split(':')[1].trim().split(' ')[0]) ?? 0;
        return kb ~/ 1024;
      }
    }
  } catch (_) {}
  return 0;
}

Uint8List _placeholderImage() {
  return Uint8List.fromList([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x00, 0x00, 0x00, 0x00, 0x3A, 0x7E, 0x9B,
    0x55, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
    0x54, 0x78, 0x9C, 0x62, 0x00, 0x00, 0x00, 0x02,
    0x00, 0x01, 0xE2, 0x21, 0xBC, 0x33, 0x00, 0x00,
    0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42,
    0x60, 0x82,
  ]);
}
