import 'dart:async';
import 'dart:convert';
import 'dart:io' if (dart.library.html) 'io_stub.dart';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:path_provider/path_provider.dart';
import '../models/chat_message.dart';

/// On-disk persistence for chat history.
/// - Conversation metadata is one small JSON file at `<docs>/conversations.json`.
/// - Image bytes live as separate files under `<docs>/chat_images/` and are
///   referenced by path. Keeping bytes out of JSON keeps writes fast and the
///   index small even with hundreds of generations.
class ConversationStore {
  ConversationStore._();
  static final ConversationStore instance = ConversationStore._();

  static const _indexFileName = 'conversations.json';
  static const _imagesDirName = 'chat_images';

  Timer? _saveDebounce;
  String? _docsPath;

  Future<String> _docs() async {
    return _docsPath ??=
        (await getApplicationDocumentsDirectory()).path;
  }

  Future<List<Conversation>> load() async {
    if (kIsWeb) return [];
    try {
      final f = File('${await _docs()}/$_indexFileName');
      if (!f.existsSync()) return [];
      final raw = await f.readAsString();
      final list = (jsonDecode(raw) as List).cast<Map<String, dynamic>>();
      return list.map(Conversation.fromJson).toList();
    } catch (_) {
      return [];
    }
  }

  /// Coalesces rapid mutations into a single write ~150ms after the last edit.
  /// Safe to call from inside notifyListeners chains.
  void scheduleSave(List<Conversation> conversations) {
    if (kIsWeb) return;
    _saveDebounce?.cancel();
    _saveDebounce = Timer(const Duration(milliseconds: 150), () {
      _writeNow(conversations);
    });
  }

  Future<void> flush(List<Conversation> conversations) async {
    if (kIsWeb) return;
    _saveDebounce?.cancel();
    _saveDebounce = null;
    await _writeNow(conversations);
  }

  Future<void> _writeNow(List<Conversation> conversations) async {
    try {
      final docs = await _docs();
      final f = File('$docs/$_indexFileName');
      final tmp = File('$docs/$_indexFileName.tmp');
      final encoded =
          jsonEncode(conversations.map((c) => c.toJson()).toList());
      await tmp.writeAsString(encoded, flush: true);
      await tmp.rename(f.path);
    } catch (_) {
      // Persistence is best-effort; never crash the UI.
    }
  }

  /// Persists [bytes] to a fresh PNG and returns the absolute path.
  Future<String?> saveImage(Uint8List bytes, {required String tag}) async {
    if (kIsWeb) return null;
    try {
      final docs = await _docs();
      final dir = Directory('$docs/$_imagesDirName');
      if (!dir.existsSync()) dir.createSync(recursive: true);
      final ts = DateTime.now().microsecondsSinceEpoch;
      final path = '${dir.path}/${tag}_$ts.png';
      await File(path).writeAsBytes(bytes, flush: false);
      return path;
    } catch (_) {
      return null;
    }
  }

  Future<void> deleteImage(String? path) async {
    if (path == null || kIsWeb) return;
    try {
      final f = File(path);
      if (f.existsSync()) await f.delete();
    } catch (_) {}
  }
}
