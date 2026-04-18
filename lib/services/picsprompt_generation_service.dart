import 'package:flutter/services.dart';
import 'picsprompt_config.dart';

/// ONNX Runtime execution provider chosen for a generation. Maps 1:1 to the
/// `executionProvider` string the native side reads off the method channel.
enum ExecutionProvider {
  cpu('cpu'),
  xnnpack('xnnpack'),
  nnapi('nnapi');

  final String wire;
  const ExecutionProvider(this.wire);
}

class GenerationRequest {
  final String prompt;
  final String negativePrompt;
  final int width;
  final int height;
  final int steps;
  final double guidance;
  final int seed;
  final String outputPath;

  /// Native scheduler: `lcm` (DreamShaper-8-LCM ONNX) or `standard_sd` (vanilla SD1.5 ONNX).
  final String pipelineKind;

  final ExecutionProvider executionProvider;

  const GenerationRequest({
    required this.prompt,
    required this.negativePrompt,
    required this.width,
    required this.height,
    required this.steps,
    required this.guidance,
    required this.seed,
    required this.outputPath,
    this.pipelineKind = 'lcm',
    this.executionProvider = ExecutionProvider.cpu,
  });

  Map<String, dynamic> toMap() => {
        'prompt': prompt,
        'negativePrompt': negativePrompt,
        'width': width,
        'height': height,
        'steps': steps,
        'guidance': guidance,
        'seed': seed,
        'outputPath': outputPath,
        'pipelineKind': pipelineKind,
        'executionProvider': executionProvider.wire,
      };
}

class PicspromptGenerationService {
  static const MethodChannel _channel = MethodChannel('picsprompt.inference');
  final ConfigBundle config;

  PicspromptGenerationService(this.config);

  Future<void> initModel({required String modelDir}) async {
    await _channel.invokeMethod('initModel', {'modelDir': modelDir});
  }

  GenerationRequest buildRequestFromTemplate({
    required String templateId,
    required String outputPath,
    int? seedOverride,
  }) {
    final template = config.templatesById[templateId];
    if (template == null) {
      throw ArgumentError('Unknown templateId: $templateId');
    }

    final profile = config.profilesById[template.profileId];
    if (profile == null) {
      throw StateError(
        'Template ${template.id} points to missing profile ${template.profileId}',
      );
    }

    return GenerationRequest(
      prompt: template.prompt,
      negativePrompt: template.negativePrompt,
      width: profile.width,
      height: profile.height,
      steps: profile.steps,
      guidance: profile.guidance,
      seed: seedOverride ?? template.defaultSeed,
      outputPath: outputPath,
    );
  }

  Future<String> generate(GenerationRequest request) async {
    final result = await _channel.invokeMethod<String>(
      'generateImage',
      request.toMap(),
    );
    if (result == null || result.isEmpty) {
      throw StateError('Native generateImage returned empty output path');
    }
    return result;
  }

  Future<String> generateAnimeGan({
    required String modelPath,
    required Uint8List imageBytes,
    required String outputPath,
    ExecutionProvider executionProvider = ExecutionProvider.cpu,
  }) async {
    final result = await _channel.invokeMethod<String>(
      'generateAnimeGan',
      {
        'modelPath': modelPath,
        'imageBytes': imageBytes,
        'outputPath': outputPath,
        'executionProvider': executionProvider.wire,
      },
    );
    if (result == null || result.isEmpty) {
      throw StateError('Native generateAnimeGan returned empty output path');
    }
    return result;
  }

  /// DreamShaper-LCM img2img — VAE encode → noise (strength) → LCM denoise →
  /// VAE decode. Reuses the bundle already loaded by [initModel].
  Future<String> generateLcmImg2Img({
    required Uint8List imageBytes,
    required String prompt,
    required String negativePrompt,
    required String outputPath,
    double strength = 0.55,
    int steps = 8,
    double guidance = 2.0,
    int width = 512,
    int height = 512,
    int? seed,
    ExecutionProvider executionProvider = ExecutionProvider.cpu,
  }) async {
    final result = await _channel.invokeMethod<String>(
      'generateLcmImg2Img',
      {
        'imageBytes': imageBytes,
        'prompt': prompt,
        'negativePrompt': negativePrompt,
        'outputPath': outputPath,
        'strength': strength,
        'steps': steps,
        'guidance': guidance,
        'width': width,
        'height': height,
        'seed': seed ?? DateTime.now().millisecondsSinceEpoch & 0x7FFFFFFF,
        'executionProvider': executionProvider.wire,
      },
    );
    if (result == null || result.isEmpty) {
      throw StateError('Native generateLcmImg2Img returned empty output path');
    }
    return result;
  }
}
