import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'inference_service.dart';

class TestLogger {
  // Singleton
  static final TestLogger _instance = TestLogger._internal();
  factory TestLogger() => _instance;
  TestLogger._internal();

  // All recorded entries this session
  final List<Map<String, dynamic>> _entries = [];

  // Called after every generation — completely silent
  // User sees nothing, data just gets recorded
  Future<void> record({
    required String prompt,
    required GenerationResult result,
    required String deviceName,
    required int batteryBefore,
    required int batteryAfter,
    required double cpuUsage,
    required double temperatureCelsius,
  }) async {
    final entry = {
      // When
      'timestamp':         DateTime.now().toIso8601String(),

      // What was generated
      'prompt':            prompt,

      // Which model
      'model':             result.modelUsed,
      'execution_provider':result.executionProvider.wire,
      'resolution':        result.resolution,
      'inference_steps':   result.inferenceSteps,

      // Timing (ms)
      'total_time_ms':     result.totalTimeMs,
      'model_load_ms':     result.modelLoadTimeMs,
      'inference_ms':      result.inferenceTimeMs,

      // Hardware
      'ram_used_mb':       result.ramUsedMB,
      'cpu_percent':       cpuUsage.toStringAsFixed(1),
      'temp_celsius':      temperatureCelsius.toStringAsFixed(1),

      // Battery
      'battery_before':    batteryBefore,
      'battery_after':     batteryAfter,
      'battery_drained':   (batteryBefore - batteryAfter)
                               .toStringAsFixed(1),

      // Device info
      'device':            deviceName,
    };

    _entries.add(entry);

    // Also save to file immediately
    // So data is not lost if app crashes
    await _saveToFile();
  }

  // Saves all entries to CSV file on device
  Future<void> _saveToFile() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/picsprompt_test_logs.csv');

      // Build CSV content
      final buffer = StringBuffer();

      // Header row — only write once
      if (!file.existsSync()) {
        buffer.writeln(_csvHeader());
      }

      // Write latest entry
      final latest = _entries.last;
      buffer.writeln(_entryToCSVRow(latest));

      // Append to file
      await file.writeAsString(
        buffer.toString(),
        mode: FileMode.append,
      );
    } catch (e) {
      // Silent fail — never crash the app for logging
    }
  }

  // Exports CSV — user taps "Export Logs" in settings
  Future<void> exportCSV() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/picsprompt_test_logs.csv');

      if (!file.existsSync()) {
        return; // nothing to export yet
      }

      // Share the file via WhatsApp, Email, Drive etc
      await Share.shareXFiles(
        [XFile(file.path)],
        subject: 'PicsPrompt Test Logs',
        text: 'PicsPrompt testing data — ${_entries.length} generations recorded',
      );
    } catch (e) {
      // handle error
    }
  }

  // How many generations recorded this session
  int get entryCount => _entries.length;

  // Clears log file — fresh start for new device test
  Future<void> clearLogs() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/picsprompt_test_logs.csv');
      if (file.existsSync()) {
        await file.delete();
      }
      _entries.clear();
    } catch (e) {
      // silent fail
    }
  }

  // CSV column headers
  String _csvHeader() {
    return [
      'timestamp',
      'prompt',
      'model',
      'execution_provider',
      'resolution',
      'inference_steps',
      'total_time_ms',
      'model_load_ms',
      'inference_ms',
      'ram_used_mb',
      'cpu_percent',
      'temp_celsius',
      'battery_before',
      'battery_after',
      'battery_drained',
      'device',
    ].join(',');
  }

  // Converts one entry map to a CSV row
  String _entryToCSVRow(Map<String, dynamic> entry) {
    return [
      entry['timestamp'],
      '"${entry['prompt']}"',   // wrap in quotes (prompt may have commas)
      entry['model'],
      entry['execution_provider'],
      entry['resolution'],
      entry['inference_steps'],
      entry['total_time_ms'],
      entry['model_load_ms'],
      entry['inference_ms'],
      entry['ram_used_mb'],
      entry['cpu_percent'],
      entry['temp_celsius'],
      entry['battery_before'],
      entry['battery_after'],
      entry['battery_drained'],
      '"${entry['device']}"',
    ].join(',');
  }
}