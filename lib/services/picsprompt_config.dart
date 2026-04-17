import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;

class TemplateConfig {
  final String id;
  final String name;
  final String prompt;
  final String negativePrompt;
  final int defaultSeed;
  final String profileId;

  const TemplateConfig({
    required this.id,
    required this.name,
    required this.prompt,
    required this.negativePrompt,
    required this.defaultSeed,
    required this.profileId,
  });

  factory TemplateConfig.fromJson(Map<String, dynamic> json) {
    return TemplateConfig(
      id: json['id'] as String,
      name: json['name'] as String,
      prompt: json['prompt'] as String,
      negativePrompt: json['negativePrompt'] as String,
      defaultSeed: json['defaultSeed'] as int,
      profileId: json['profileId'] as String,
    );
  }
}

class ProfileConfig {
  final String id;
  final int width;
  final int height;
  final int steps;
  final double guidance;

  const ProfileConfig({
    required this.id,
    required this.width,
    required this.height,
    required this.steps,
    required this.guidance,
  });

  factory ProfileConfig.fromJson(Map<String, dynamic> json) {
    return ProfileConfig(
      id: json['id'] as String,
      width: json['width'] as int,
      height: json['height'] as int,
      steps: json['steps'] as int,
      guidance: (json['guidance'] as num).toDouble(),
    );
  }
}

class ConfigBundle {
  final Map<String, TemplateConfig> templatesById;
  final Map<String, ProfileConfig> profilesById;

  const ConfigBundle({
    required this.templatesById,
    required this.profilesById,
  });
}

class PicspromptConfigLoader {
  static const String templatesAsset = 'assets/config/templates.json';
  static const String profilesAsset = 'assets/config/profiles.json';

  Future<ConfigBundle> load() async {
    final templatesRaw = await rootBundle.loadString(templatesAsset);
    final profilesRaw = await rootBundle.loadString(profilesAsset);

    final templatesJson = jsonDecode(templatesRaw) as Map<String, dynamic>;
    final profilesJson = jsonDecode(profilesRaw) as Map<String, dynamic>;

    final templates = (templatesJson['templates'] as List<dynamic>)
        .map((e) => TemplateConfig.fromJson(e as Map<String, dynamic>))
        .toList();
    final profiles = (profilesJson['profiles'] as List<dynamic>)
        .map((e) => ProfileConfig.fromJson(e as Map<String, dynamic>))
        .toList();

    return ConfigBundle(
      templatesById: {for (final t in templates) t.id: t},
      profilesById: {for (final p in profiles) p.id: p},
    );
  }
}
