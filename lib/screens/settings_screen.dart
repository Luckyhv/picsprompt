import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';
import '../services/benchmark_runner.dart';
import '../services/picsprompt_generation_service.dart';
import '../services/test_logger.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _benchRunning = false;
  String? _benchStatus;

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _buildSection(
          context,
          title: 'Generation',
          children: [
            _buildSettingTile(
              icon: Icons.straighten,
              title: 'Image Size',
              subtitle: '256 x 256',
            ),
            _buildSettingTile(
              icon: Icons.repeat,
              title: 'Inference Steps',
              subtitle:
                  'DreamShaper (LCM): 12–16 steps by profile; chat uses 14 at 256px (faster presets use 12).',
            ),
            _buildSettingTile(
              icon: Icons.tune,
              title: 'Guidance Scale',
              subtitle: 'LCM uses ~1.8–2.2 (not SD 7.5). Set in assets/config/profiles.json.',
            ),
          ],
        ),
        const SizedBox(height: 24),
        _buildSection(
          context,
          title: 'Storage',
          children: [
            _buildSettingTile(
              icon: Icons.folder_outlined,
              title: 'Model Storage',
              subtitle:
                  'DreamShaper ONNX is pushed via adb (not in APK). Typical bundle: ~1 GB (quint8/int8) or ~3 GB (int8_mobile).',
            ),
            _buildSettingTile(
              icon: Icons.delete_outline,
              title: 'Clear Generated Images',
              subtitle: 'Free up space',
              onTap: () => _confirmClearImages(context),
            ),
          ],
        ),
        const SizedBox(height: 24),
        _buildSection(
          context,
          title: 'Research & Testing',
          children: [
            _buildSettingTile(
              icon: Icons.upload_outlined,
              title: 'Export Test Logs',
              subtitle: 'Share CSV file for research paper',
              onTap: () => _exportLogs(context),
            ),
            _buildSettingTile(
              icon: Icons.delete_sweep_outlined,
              title: 'Clear Test Logs',
              subtitle: 'Start fresh for new device test',
              onTap: () => _confirmClearLogs(context),
            ),
          ],
        ),
        const SizedBox(height: 24),
        _buildSection(
          context,
          title: 'Benchmark — AnimeGAN (1 run per tap, ~3-8 s)',
          children: [
            _buildSettingTile(
              icon: _benchRunning ? Icons.hourglass_top : Icons.memory,
              title: 'AnimeGAN — CPU',
              subtitle: 'Single run, no warmup',
              onTap: _benchRunning
                  ? null
                  : () => _runBenchmark([ExecutionProvider.cpu]),
            ),
            _buildSettingTile(
              icon: _benchRunning ? Icons.hourglass_top : Icons.bolt,
              title: 'AnimeGAN — XNNPACK',
              subtitle: 'CPU SIMD kernels',
              onTap: _benchRunning
                  ? null
                  : () => _runBenchmark([ExecutionProvider.xnnpack]),
            ),
            _buildSettingTile(
              icon: _benchRunning ? Icons.hourglass_top : Icons.developer_board,
              title: 'AnimeGAN — NNAPI',
              subtitle: 'MediaTek APU/GPU path (may crash)',
              onTap: _benchRunning
                  ? null
                  : () => _runBenchmark([ExecutionProvider.nnapi]),
            ),
          ],
        ),
        const SizedBox(height: 24),
        _buildSection(
          context,
          title: 'Benchmark — Butterflies DDPM (~30-60 s/iter)',
          children: [
            _buildSettingTile(
              icon: _benchRunning ? Icons.hourglass_top : Icons.memory,
              title: 'Butterflies — CPU',
              subtitle: 'Single run, no warmup',
              onTap: _benchRunning
                  ? null
                  : () => _runBenchmark(
                        [ExecutionProvider.cpu],
                        modelId: 'butterflies',
                        iterations: 1,
                        warmup: 0,
                        cooldownPerIter: const Duration(seconds: 30),
                        cooldownPerEp: const Duration(seconds: 60),
                      ),
            ),
            _buildSettingTile(
              icon: _benchRunning ? Icons.hourglass_top : Icons.bolt,
              title: 'Butterflies — XNNPACK',
              subtitle: 'CPU SIMD kernels',
              onTap: _benchRunning
                  ? null
                  : () => _runBenchmark(
                        [ExecutionProvider.xnnpack],
                        modelId: 'butterflies',
                        iterations: 1,
                        warmup: 0,
                        cooldownPerIter: const Duration(seconds: 30),
                      ),
            ),
            _buildSettingTile(
              icon: _benchRunning ? Icons.hourglass_top : Icons.developer_board,
              title: 'Butterflies — NNAPI',
              subtitle: 'High crash risk on UNet — try last',
              onTap: _benchRunning
                  ? null
                  : () => _runBenchmark(
                        [ExecutionProvider.nnapi],
                        modelId: 'butterflies',
                        iterations: 1,
                        warmup: 0,
                        cooldownPerIter: const Duration(seconds: 30),
                      ),
            ),
          ],
        ),
        const SizedBox(height: 24),
        _buildSection(
          context,
          title: 'Benchmark — DreamShaper LCM (~30-90 s/iter)',
          children: [
            _buildSettingTile(
              icon: _benchRunning ? Icons.hourglass_top : Icons.memory,
              title: 'DreamShaper — CPU',
              subtitle: 'Single run at 256², no warmup',
              onTap: _benchRunning
                  ? null
                  : () => _runBenchmark(
                        [ExecutionProvider.cpu],
                        modelId: 'lcm',
                        iterations: 1,
                        warmup: 0,
                        cooldownPerIter: const Duration(seconds: 30),
                        cooldownPerEp: const Duration(seconds: 60),
                      ),
            ),
            _buildSettingTile(
              icon: _benchRunning ? Icons.hourglass_top : Icons.bolt,
              title: 'DreamShaper — XNNPACK',
              subtitle: 'CPU SIMD kernels',
              onTap: _benchRunning
                  ? null
                  : () => _runBenchmark(
                        [ExecutionProvider.xnnpack],
                        modelId: 'lcm',
                        iterations: 1,
                        warmup: 0,
                        cooldownPerIter: const Duration(seconds: 30),
                      ),
            ),
            _buildSettingTile(
              icon: _benchRunning ? Icons.hourglass_top : Icons.developer_board,
              title: 'DreamShaper — NNAPI',
              subtitle: 'Likely to crash on a Dimensity 920 — skip if unstable',
              onTap: _benchRunning
                  ? null
                  : () => _runBenchmark(
                        [ExecutionProvider.nnapi],
                        modelId: 'lcm',
                        iterations: 1,
                        warmup: 0,
                        cooldownPerIter: const Duration(seconds: 30),
                      ),
            ),
            _buildSettingTile(
              icon: Icons.info_outline,
              title: _benchRunning ? 'Running…' : 'Status',
              subtitle: _benchStatus ??
                  'Run scripts/bench_android.py before each tile to capture system metrics.',
            ),
          ],
        ),
        const SizedBox(height: 24),
        _buildSection(
          context,
          title: 'About',
          children: [
            _buildSettingTile(
              icon: Icons.info_outline,
              title: 'PicsPrompt',
              subtitle: 'v1.0.0 • IFT 593 Project',
            ),
            _buildSettingTile(
              icon: Icons.school_outlined,
              title: 'Arizona State University',
              subtitle: 'On-device AI Image Generation',
            ),
          ],
        ),
      ],
    );
  }

  // Export CSV via share sheet
  Future<void> _exportLogs(BuildContext context) async {
    final count = TestLogger().entryCount;

    if (count == 0) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text('No logs yet. Generate some images first.'),
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
        ),
      );
      return;
    }

    await TestLogger().exportCSV();
  }

  // Confirm before clearing logs
  void _confirmClearLogs(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Clear Test Logs?'),
        content: const Text(
          'This will delete all recorded test data. '
          'Do this before testing on a new device.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              await TestLogger().clearLogs();
              if (!ctx.mounted) return;
              Navigator.pop(ctx);
              if (!context.mounted) return;
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: const Text('Test logs cleared'),
                  behavior: SnackBarBehavior.floating,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
              );
            },
            child: const Text(
              'Clear',
              style: TextStyle(color: Colors.red),
            ),
          ),
        ],
      ),
    );
  }

  // Run AnimeGAN across all three EPs and share the JSONL via the system sheet.
  // Butterflies / DreamShaper are skipped here because each takes minutes on a
  // mid-range device — add them back in BenchmarkSpec once a small run is green.
  Future<void> _runBenchmark(
    List<ExecutionProvider> providers, {
    String modelId = 'animegan',
    int iterations = 1,
    int warmup = 0,
    Duration cooldownPerIter = const Duration(seconds: 8),
    Duration cooldownPerEp = const Duration(seconds: 20),
  }) async {
    setState(() {
      _benchRunning = true;
      _benchStatus = 'Starting $modelId…';
    });
    String? outPath;
    String? error;
    try {
      outPath = await BenchmarkRunner().run(specs: [
        BenchmarkSpec(
          modelId: modelId,
          providers: providers,
          iterations: iterations,
          warmup: warmup,
          cooldownPerIter: cooldownPerIter,
          cooldownPerEp: cooldownPerEp,
        ),
      ]);
    } catch (e) {
      error = e.toString();
    }
    if (!mounted) return;
    setState(() {
      _benchRunning = false;
      _benchStatus = error ?? 'Last run: ${outPath ?? "(unknown)"}';
    });
    if (error != null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Benchmark failed: $error')),
      );
      return;
    }
    if (outPath != null) {
      await Share.shareXFiles(
        [XFile(outPath)],
        text: 'PicsPrompt benchmark CSV',
      );
    }
  }

  // Confirm before clearing images
  void _confirmClearImages(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Clear Generated Images?'),
        content: const Text(
          'This will remove all generated images from conversations.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(ctx);
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: const Text('Images cleared'),
                  behavior: SnackBarBehavior.floating,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
              );
            },
            child: const Text(
              'Clear',
              style: TextStyle(color: Colors.red),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSection(
    BuildContext context, {
    required String title,
    required List<Widget> children,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
        ),
        const SizedBox(height: 12),
        Container(
          decoration: BoxDecoration(
            border: Border.all(color: Colors.grey.shade200),
            borderRadius: BorderRadius.circular(16),
          ),
          child: Column(
            children: List.generate(children.length, (i) {
              return Column(
                children: [
                  children[i],
                  if (i < children.length - 1)
                    Divider(
                      height: 1,
                      indent: 56,
                      color: Colors.grey.shade200,
                    ),
                ],
              );
            }),
          ),
        ),
      ],
    );
  }

  // Added onTap parameter
  Widget _buildSettingTile({
    required IconData icon,
    required String title,
    required String subtitle,
    VoidCallback? onTap,
  }) {
    return ListTile(
      onTap: onTap,
      leading: Icon(icon, color: Colors.grey.shade600),
      title: Text(title, style: const TextStyle(fontSize: 14)),
      subtitle: Text(subtitle, style: const TextStyle(fontSize: 12)),
      contentPadding: const EdgeInsets.symmetric(horizontal: 16),
      trailing: onTap != null
          ? Icon(Icons.chevron_right, color: Colors.grey.shade400, size: 18)
          : null,
    );
  }
}
