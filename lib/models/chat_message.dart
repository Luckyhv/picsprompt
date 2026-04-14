import 'dart:typed_data';

class ChatMessage {
  ChatMessage({
    required this.text,
    required this.isUser,
    this.imageBytes,
    this.imagePath,
    this.isGenerating = false,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  final String text;
  final bool isUser;
  Uint8List? imageBytes;
  String? imagePath; // disk-backed copy of imageBytes
  final bool isGenerating;
  final DateTime timestamp;

  Map<String, dynamic> toJson() => {
        'text': text,
        'isUser': isUser,
        'imagePath': imagePath,
        'ts': timestamp.millisecondsSinceEpoch,
      };

  static ChatMessage fromJson(Map<String, dynamic> j) => ChatMessage(
        text: j['text'] as String? ?? '',
        isUser: j['isUser'] as bool? ?? false,
        imagePath: j['imagePath'] as String?,
        timestamp: DateTime.fromMillisecondsSinceEpoch(
          j['ts'] as int? ?? DateTime.now().millisecondsSinceEpoch,
        ),
      );
}

class Conversation {
  Conversation({
    required this.id,
    required this.title,
    required this.modelId,
    List<ChatMessage>? messages,
    DateTime? createdAt,
  })  : messages = messages ?? [],
        createdAt = createdAt ?? DateTime.now();

  final String id;
  String title;
  final String modelId;
  final List<ChatMessage> messages;
  final DateTime createdAt;

  Map<String, dynamic> toJson() => {
        'id': id,
        'title': title,
        'modelId': modelId,
        'createdAt': createdAt.millisecondsSinceEpoch,
        'messages': messages
            .where((m) => !m.isGenerating)
            .map((m) => m.toJson())
            .toList(),
      };

  static Conversation fromJson(Map<String, dynamic> j) => Conversation(
        id: j['id'] as String,
        title: j['title'] as String? ?? 'Conversation',
        modelId: j['modelId'] as String? ?? 'lcm',
        createdAt: DateTime.fromMillisecondsSinceEpoch(
          j['createdAt'] as int? ?? DateTime.now().millisecondsSinceEpoch,
        ),
        messages: ((j['messages'] as List?) ?? const [])
            .cast<Map<String, dynamic>>()
            .map(ChatMessage.fromJson)
            .toList(),
      );
}

class AIModel {
  const AIModel({
    required this.id,
    required this.name,
    required this.description,
    required this.size,
    required this.ramRequiredMB,
    this.isDownloaded = false,
  });

  final String id;
  final String name;
  final String description;
  final String size;
  final int ramRequiredMB;
  final bool isDownloaded;
}
