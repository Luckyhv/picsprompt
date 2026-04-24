import 'dart:io' show File;
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:battery_plus/battery_plus.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:image_picker/image_picker.dart';
import 'package:thermal/thermal.dart';
import '../models/app_state.dart';
import '../models/chat_message.dart';
import '../services/inference_service.dart';
import '../services/picsprompt_generation_service.dart' show ExecutionProvider;
import '../services/test_logger.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final Battery _battery = Battery();
  final ImagePicker _picker = ImagePicker();
  Uint8List? _attachedImageBytes;
  String? _attachedImageName;

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _pickImage() async {
    try {
      final picked = await _picker.pickImage(
        source: ImageSource.gallery,
        // Keep the native side's center-crop+resize fed with detail; we
        // resize down to 512 ourselves with a higher-quality multi-step
        // path, so a bigger source helps quality.
        maxWidth: 4096,
        maxHeight: 4096,
        imageQuality: 100,
      );
      if (picked == null) return;
      final bytes = await picked.readAsBytes();
      setState(() {
        _attachedImageBytes = bytes;
        _attachedImageName = picked.name;
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not open image: $e')),
      );
    }
  }

  void _clearAttachedImage() {
    setState(() {
      _attachedImageBytes = null;
      _attachedImageName = null;
    });
  }

  Future<void> _sendMessage() async {
    final appState = context.read<AppState>();
    final isUnconditional = appState.selectedModelId == 'butterflies';
    final isAnimeGan = appState.selectedModelId == 'animegan';
    final isLcmAvatar = appState.selectedModelId == 'lcm_avatar';
    final isImg2Img = isAnimeGan || isLcmAvatar;
    final text = _controller.text.trim();
    if (isImg2Img && _attachedImageBytes == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'Tap the image icon to pick a photo for the avatar.',
          ),
        ),
      );
      return;
    }
    if (text.isEmpty && !isUnconditional && !isImg2Img) return;
    final promptForLog = isImg2Img
        ? (text.isEmpty
            ? '(avatar from ${_attachedImageName ?? "image"})'
            : '$text [avatar from ${_attachedImageName ?? "image"}]')
        : (text.isEmpty ? '(unconditional sample)' : text);

    final attachedForCall = _attachedImageBytes;
    final userMessageBytes = isImg2Img ? _attachedImageBytes : null;

    _controller.clear();

    await appState.addMessage(
      ChatMessage(
        text: promptForLog,
        isUser: true,
        imageBytes: userMessageBytes,
      ),
    );
    _scrollToBottom();

    if (isImg2Img) {
      _clearAttachedImage();
    }

    await appState.addMessage(ChatMessage(
      text: isImg2Img ? 'Stylizing your photo...' : 'Generating your image...',
      isUser: false,
      isGenerating: true,
    ));
    _scrollToBottom();

    try {
      final batteryBefore = await _battery.batteryLevel;
      final deviceName = await _getDeviceName();

      // Route chat generations through NNAPI so ORT can offload to the
      // device GPU/NPU/DSP. Native sessionOpts() falls back to CPU per node
      // if a kernel can't be compiled, so this is safe across SoCs.
      const chatEp = ExecutionProvider.nnapi;
      final GenerationResult result;
      if (isLcmAvatar) {
        result = await InferenceService().generateLcmAvatar(
          inputImageBytes: attachedForCall!,
          stylePrompt: text,
          executionProvider: chatEp,
        );
      } else if (isAnimeGan) {
        result = await InferenceService().generateAnimeganAvatar(
          inputImageBytes: attachedForCall!,
          executionProvider: chatEp,
        );
      } else {
        result = await InferenceService().generate(
          prompt: text,
          modelId: appState.selectedModelId,
          // 50 is the standard DDPM demo budget; LCM defaults to 14 inside the service.
          steps: isUnconditional ? 50 : 14,
          executionProvider: chatEp,
        );
      }

      final batteryAfter = await _battery.batteryLevel;

      await TestLogger().record(
        prompt: promptForLog,
        result: result,
        deviceName: deviceName,
        batteryBefore: batteryBefore,
        batteryAfter: batteryAfter,
        cpuUsage: 0.0,
        temperatureCelsius: await _getTemperature(),
      );

      appState.removeLastMessageTransient();
      if (result.modelUsed.endsWith(':no-onnx-bundle')) {
        final String modelLabel;
        final String pushScript;
        if (result.modelUsed.startsWith('butterflies:')) {
          modelLabel = 'Butterflies DDPM';
          pushScript = './scripts/push_butterflies_to_android.sh';
        } else if (result.modelUsed.startsWith('animegan:')) {
          modelLabel = 'AnimeGAN avatar';
          pushScript = './scripts/push_animegan_to_android.sh';
        } else if (result.modelUsed.startsWith('lcm_avatar:')) {
          modelLabel = 'AI Avatar (DreamShaper)';
          pushScript = './scripts/push_dreamshaper_to_android.sh';
        } else {
          modelLabel = 'DreamShaper';
          pushScript = './scripts/push_dreamshaper_to_android.sh';
        }
        await appState.addMessage(ChatMessage(
          text:
              '$modelLabel ONNX is not on this phone yet. On your computer run:\n'
              'cd picsprompt-models && $pushScript\n'
              'Then restart the app.',
          isUser: false,
        ));
      } else {
        await appState.addMessage(ChatMessage(
          text: promptForLog,
          isUser: false,
          imageBytes: result.imageBytes,
        ));
      }
      _scrollToBottom();
    } catch (e) {
      appState.removeLastMessageTransient();
      final detail = e is StateError ? e.message : e.toString();
      final short =
          detail.length > 200 ? '${detail.substring(0, 200)}…' : detail;
      await appState.addMessage(ChatMessage(
        text: 'Generation failed.\n$short',
        isUser: false,
      ));
      _scrollToBottom();
    }
  }

  Future<String> _getDeviceName() async {
    try {
      final info = DeviceInfoPlugin();
      if (Theme.of(context).platform == TargetPlatform.android) {
        final android = await info.androidInfo;
        return '${android.brand}_${android.model}';
      } else {
        final ios = await info.iosInfo;
        return ios.utsname.machine;
      }
    } catch (_) {
      return 'unknown_device';
    }
  }

  Future<double> _getTemperature() async {
    try {
      final temp = await Thermal().onThermalStatusChanged.first;
      return temp.index.toDouble();
    } catch (_) {
      return 0.0;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<AppState>(
      builder: (context, appState, _) {
        final messages = appState.activeConversation?.messages ?? const [];
        return Column(
          children: [
            Expanded(
              child: messages.isEmpty
                  ? _buildEmptyState(context)
                  : _buildMessageList(messages),
            ),
            _buildInputBar(context),
          ],
        );
      },
    );
  }

  Widget _buildEmptyState(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Icon(
                Icons.auto_awesome,
                size: 40,
                color: Colors.grey.shade600,
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Start Creating',
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
            ),
            const SizedBox(height: 8),
            Text(
              'Type a prompt to generate an image with DreamShaper.',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.grey.shade600,
                  ),
            ),
            const SizedBox(height: 16),
            Text(
              'All processing happens on your device.',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: Colors.grey.shade400,
                  ),
            ),
          ],
        ),
      ),
    );
  }

  // Prefers in-memory bytes (fresh generation) over the on-disk copy.
  // Image.file is decoded lazily; thumbnails for older messages don't pay until scrolled into view.
  Widget _imageWidget({
    Uint8List? bytes,
    String? path,
    double? height,
    BoxFit fit = BoxFit.contain,
  }) {
    if (bytes != null) {
      return Image.memory(bytes, height: height, fit: fit, gaplessPlayback: true);
    }
    if (path != null && !kIsWeb) {
      return Image.file(File(path), height: height, fit: fit, gaplessPlayback: true);
    }
    return SizedBox(height: height ?? 0);
  }

  Widget _buildMessageList(List<ChatMessage> messages) {
    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      itemCount: messages.length,
      itemBuilder: (context, index) {
        return _buildMessageBubble(context, messages[index]);
      },
    );
  }

  Widget _buildMessageBubble(BuildContext context, ChatMessage msg) {
    final isUser = msg.isUser;
    final colorScheme = Theme.of(context).colorScheme;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.78,
        ),
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: isUser
              ? colorScheme.primary
              : colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(16),
            topRight: const Radius.circular(16),
            bottomLeft: Radius.circular(isUser ? 16 : 4),
            bottomRight: Radius.circular(isUser ? 4 : 16),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (msg.isGenerating)
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  SizedBox(
                    width: 14,
                    height: 14,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: colorScheme.onSurfaceVariant,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    msg.text,
                    style: TextStyle(color: colorScheme.onSurfaceVariant),
                  ),
                ],
              )
            else ...[
              Text(
                msg.text,
                style: TextStyle(
                  color: isUser
                      ? colorScheme.onPrimary
                      : colorScheme.onSurface,
                ),
              ),
              if (msg.imageBytes != null || msg.imagePath != null) ...[
                const SizedBox(height: 8),
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: _imageWidget(
                    bytes: msg.imageBytes,
                    path: msg.imagePath,
                    fit: BoxFit.contain,
                  ),
                ),
              ],
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildInputBar(BuildContext context) {
    final selectedId = context.watch<AppState>().selectedModelId;
    final isUnconditional = selectedId == 'butterflies';
    final isImg2Img = selectedId == 'animegan' || selectedId == 'lcm_avatar';
    final isLcmAvatar = selectedId == 'lcm_avatar';
    final colorScheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.fromLTRB(8, 8, 8, 8),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        border: Border(top: BorderSide(color: Colors.grey.shade200)),
      ),
      child: SafeArea(
        top: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (isImg2Img && _attachedImageBytes != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  children: [
                    ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.memory(
                        _attachedImageBytes!,
                        width: 40,
                        height: 40,
                        fit: BoxFit.cover,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _attachedImageName ?? 'attached image',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey.shade700,
                        ),
                      ),
                    ),
                    IconButton(
                      visualDensity: VisualDensity.compact,
                      icon: const Icon(Icons.close, size: 18),
                      onPressed: _clearAttachedImage,
                    ),
                  ],
                ),
              ),
            Row(
              children: [
                if (isImg2Img)
                  IconButton(
                    icon: Icon(
                      Icons.image_outlined,
                      color: colorScheme.primary,
                    ),
                    tooltip: 'Pick a photo',
                    onPressed: _pickImage,
                  ),
                Expanded(
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.grey.shade100,
                      borderRadius: BorderRadius.circular(24),
                    ),
                    child: TextField(
                      controller: _controller,
                      decoration: InputDecoration(
                        hintText: isUnconditional
                            ? 'Tap send to sample a butterfly...'
                            : isImg2Img
                                ? (_attachedImageBytes == null
                                    ? 'Pick a photo to make an avatar'
                                    : isLcmAvatar
                                        ? 'Optional style: anime, oil painting…'
                                        : 'Tap send to stylize')
                                : 'Describe your image...',
                        border: InputBorder.none,
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 20,
                          vertical: 12,
                        ),
                      ),
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _sendMessage(),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Container(
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(
                      color: colorScheme.primary,
                      width: 1.5,
                    ),
                  ),
                  child: IconButton(
                    icon: Icon(
                      Icons.send_rounded,
                      color: colorScheme.primary,
                    ),
                    onPressed: _sendMessage,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
