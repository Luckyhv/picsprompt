// Web stub — replaces dart:io on Flutter web so the app compiles.
// These are never actually called (guarded by kIsWeb checks at runtime).

class File {
  File(this.path);
  final String path;
  bool existsSync() => false;
  int lengthSync() => 0;
  Future<void> delete() async {}
  Future<List<int>> readAsBytes() async => [];
  Future<String> readAsString() async => '';
  List<String> readAsLinesSync() => [];
  Future<void> writeAsString(String contents, {bool flush = false}) async {}
  Future<void> writeAsBytes(List<int> bytes, {bool flush = false}) async {}
  Future<File> rename(String newPath) async => File(newPath);
  File get parent => this;
  Uri get uri => Uri.file(path);
}

class Directory {
  Directory(this.path);
  final String path;
  bool existsSync() => false;
  void createSync({bool recursive = false}) {}
}

class ProcessResult {
  final String stdout = '';
  final String stderr = '';
  final int exitCode = 0;
}

class Process {
  static Future<ProcessResult> run(String exe, List<String> args) async =>
      ProcessResult();
}

// ignore: non_constant_identifier_names
abstract class Platform {
  static bool get isAndroid => false;
  static bool get isIOS => false;
}

void sleep(Duration duration) {}
