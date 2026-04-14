import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:io' if (dart.library.html) '../services/io_stub.dart';
import 'chat_message.dart';
import '../services/inference_service.dart';
import '../services/conversation_store.dart';

class AppState extends ChangeNotifier {
  // Available models. Add a model here only when its bundle path and *_OnnxReady
  // flag are wired up in InferenceService.
  final List<AIModel> _models = const [
    AIModel(
      id: 'lcm',
      name: 'DreamShaper (ONNX)',
      description:
          'Text-to-image. Native ONNX on Android. Push weights via '
          'picsprompt-models/scripts/push_dreamshaper_to_android.sh.',
      size: '~1–4 GB on device',
      ramRequiredMB: 1800,
      isDownloaded: true,
    ),
    AIModel(
      id: 'butterflies',
      name: 'Butterflies DDPM (128)',
      description:
          'Unconditional generator trained on the Smithsonian Butterflies dataset. '
          'Push via picsprompt-models/scripts/push_butterflies_to_android.sh.',
      size: '~450 MB on device',
      ramRequiredMB: 800,
      isDownloaded: true,
    ),
    AIModel(
      id: 'lcm_avatar',
      name: 'AI Avatar (DreamShaper img2img)',
      description:
          'Photo → stylized avatar via diffusion img2img on the device. '
          'Reuses the DreamShaper-LCM bundle. Optional style prompt in chat '
          '(e.g. "anime", "oil painting", "cyberpunk").',
      size: 'Reuses ~1 GB DreamShaper bundle',
      ramRequiredMB: 1800,
      isDownloaded: true,
    ),
    AIModel(
      id: 'animegan',
      name: 'AnimeGAN (legacy img2img)',
      description:
          'Older single-pass GAN. Kept for comparison; the DreamShaper avatar above '
          'produces noticeably better results. Push AnimeGAN via '
          'picsprompt-models/scripts/push_animegan_to_android.sh.',
      size: '~9 MB on device',
      ramRequiredMB: 200,
      isDownloaded: true,
    ),
  ];

  final List<Conversation> _conversations = [];
  Conversation? _activeConversation;
  String _selectedModelId = 'lcm';
  int _availableRAMmb = 0;
  bool _dreamshaperOnnxReady = false;
  bool _butterfliesOnnxReady = false;
  bool _animeganOnnxReady = false;
  bool _isLoadingModel = false;
  String? _loadingStatus;

  List<AIModel> get models => _models;
  bool get dreamshaperOnnxReady => _dreamshaperOnnxReady;
  bool get butterfliesOnnxReady => _butterfliesOnnxReady;
  bool get animeganOnnxReady => _animeganOnnxReady;

  bool isModelReady(String modelId) => switch (modelId) {
        'lcm' => _dreamshaperOnnxReady,
        'lcm_avatar' => _dreamshaperOnnxReady,
        'butterflies' => _butterfliesOnnxReady,
        'animegan' => _animeganOnnxReady,
        _ => false,
      };
  List<Conversation> get conversations => _conversations;
  Conversation? get activeConversation => _activeConversation;
  String get selectedModelId => _selectedModelId;
  int get availableRAMmb => _availableRAMmb;
  bool get isLoadingModel => _isLoadingModel;
  String? get loadingStatus => _loadingStatus;

  AIModel get selectedModel =>
      _models.firstWhere((m) => m.id == _selectedModelId);

  Future<void> initializeApp() async {
    await _checkAvailableRAM();
    if (!kIsWeb) {
      try {
        _dreamshaperOnnxReady =
            await InferenceService.hasCompleteDreamshaperBundle();
        _butterfliesOnnxReady =
            await InferenceService.hasCompleteButterfliesBundle();
        _animeganOnnxReady = await InferenceService.hasAnimeganModel();
      } catch (_) {
        _dreamshaperOnnxReady = false;
        _butterfliesOnnxReady = false;
        _animeganOnnxReady = false;
      }
    }
    final saved = await ConversationStore.instance.load();
    _conversations
      ..clear()
      ..addAll(saved);
    notifyListeners();
  }

  Future<void> _checkAvailableRAM() async {
    if (kIsWeb) return;
    try {
      if (Platform.isAndroid) {
        final result = await Process.run('cat', ['/proc/meminfo']);
        final lines = result.stdout.toString().split('\n');
        for (final line in lines) {
          if (line.startsWith('MemAvailable')) {
            final kb =
                int.tryParse(line.split(':')[1].trim().split(' ')[0]) ?? 0;
            _availableRAMmb = kb ~/ 1024;
            break;
          }
        }
      } else if (Platform.isIOS) {
        _availableRAMmb = 600;
      }
    } catch (_) {
      _availableRAMmb = 500;
    }
  }

  void selectModel(String modelId) {
    _selectedModelId = modelId;
    notifyListeners();
  }

  void setLoadingModel(bool loading, [String? status]) {
    _isLoadingModel = loading;
    _loadingStatus = status;
    notifyListeners();
  }

  /// Creates a draft conversation set as active but NOT yet inserted into the
  /// conversations list or persisted. The draft is committed on the first
  /// `addMessage` call, so abandoning a fresh chat without sending leaves no
  /// empty "New Conversation" tile behind.
  Conversation createConversation() {
    final conv = Conversation(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      title: 'New Conversation',
      modelId: _selectedModelId,
    );
    _activeConversation = conv;
    notifyListeners();
    return conv;
  }

  void setActiveConversation(Conversation? conv) {
    _activeConversation = conv;
    notifyListeners();
  }

  Future<void> addMessage(ChatMessage message) async {
    if (_activeConversation == null) {
      createConversation();
    }
    final conv = _activeConversation!;

    // Commit a freshly-created draft into the persisted list on its first
    // message. See [createConversation] for why drafts aren't inserted up front.
    if (!_conversations.any((c) => c.id == conv.id)) {
      _conversations.insert(0, conv);
    }

    if (message.imageBytes != null && message.imagePath == null) {
      message.imagePath = await ConversationStore.instance
          .saveImage(message.imageBytes!, tag: 'gen');
    }

    conv.messages.add(message);
    if (conv.messages.where((m) => m.isUser).length == 1 &&
        message.isUser &&
        message.text.isNotEmpty) {
      conv.title = message.text.length > 30
          ? '${message.text.substring(0, 30)}...'
          : message.text;
    }
    _persist();
    notifyListeners();
  }

  /// Removes the last message without persisting (used to drop transient
  /// "Generating..." placeholders before appending the real reply).
  void removeLastMessageTransient() {
    final conv = _activeConversation;
    if (conv == null || conv.messages.isEmpty) return;
    conv.messages.removeLast();
    notifyListeners();
  }

  void deleteConversation(String id) {
    final idx = _conversations.indexWhere((c) => c.id == id);
    if (idx == -1) return;
    final removed = _conversations.removeAt(idx);
    for (final m in removed.messages) {
      ConversationStore.instance.deleteImage(m.imagePath);
    }
    if (_activeConversation?.id == id) {
      _activeConversation = null;
    }
    _persist();
    notifyListeners();
  }

  void _persist() {
    ConversationStore.instance.scheduleSave(_conversations);
  }
}
