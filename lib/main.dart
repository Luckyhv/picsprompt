import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'models/app_state.dart';
import 'models/chat_message.dart';
import 'screens/chat_screen.dart';
import 'screens/chats_screen.dart';
import 'screens/home_screen.dart';
import 'screens/models_screen.dart';
import 'screens/settings_screen.dart';
import 'services/benchmark_runner.dart';
import 'services/picsprompt_generation_service.dart' show ExecutionProvider;

/// Method channel that lets `MainActivity` kick off a benchmark when the app
/// is launched with `bench_*` intent extras (i.e. from the Mac-side
/// `run_remote_bench.sh` over adb). Registered before runApp so it's live the
/// moment the engine attaches.
void _registerRemoteBenchChannel() {
  const channel = MethodChannel('picsprompt.bench');
  channel.setMethodCallHandler((call) async {
    if (call.method != 'runBenchmark') return null;
    final args = (call.arguments as Map?)?.cast<String, dynamic>() ?? {};
    final modelId = (args['model'] as String?) ?? 'animegan';
    final epWire = (args['ep'] as String?)?.toLowerCase() ?? 'cpu';
    final iterations = (args['iters'] as int?) ?? 1;
    final warmup = (args['warmup'] as int?) ?? 0;
    final ep = ExecutionProvider.values.firstWhere(
      (e) => e.wire == epWire,
      orElse: () => ExecutionProvider.cpu,
    );
    final path = await BenchmarkRunner().run(specs: [
      BenchmarkSpec(
        modelId: modelId,
        providers: [ep],
        iterations: iterations,
        warmup: warmup,
      ),
    ]);
    return path;
  });
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  _registerRemoteBenchChannel();

  final appState = AppState();
  await appState.initializeApp();

  runApp(
    ChangeNotifierProvider<AppState>(
      create: (_) => appState,
      child: const PicsPrompt(),
    ),
  );
}

class PicsPrompt extends StatelessWidget {
  const PicsPrompt({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'PicsPrompt',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF00897B), // teal-green like the reference
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        scaffoldBackgroundColor: Colors.white,
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.white,
          surfaceTintColor: Colors.transparent,
          elevation: 0,
        ),
      ),
      home: const AppShell(),
    );
  }
}

class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _currentIndex = 0;
  bool _inChatView = false;

  void _openChat([Conversation? conv]) {
    final appState = context.read<AppState>();
    if (conv != null) {
      appState.setActiveConversation(conv);
    } else {
      appState.createConversation();
    }
    setState(() {
      _currentIndex = 1;
      _inChatView = true;
    });
  }

  void _backFromChat() {
    setState(() {
      _inChatView = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: _buildAppBar(context),
      body: _buildBody(),
      bottomNavigationBar: _buildBottomNav(context),
    );
  }

  PreferredSizeWidget _buildAppBar(BuildContext context) {
    if (_currentIndex == 1 && _inChatView) {
      return _buildChatAppBar(context);
    }

    final titles = ['Home', 'Chats', 'Models', 'Settings'];
    return AppBar(
      title: Text(titles[_currentIndex]),
      centerTitle: true,
      bottom: _currentIndex == 1
          ? PreferredSize(
              preferredSize: const Size.fromHeight(0),
              child: Container(),
            )
          : null,
    );
  }

  PreferredSizeWidget _buildChatAppBar(BuildContext context) {
    return AppBar(
      leading: IconButton(
        icon: const Icon(Icons.arrow_back),
        onPressed: _backFromChat,
      ),
      title: Consumer<AppState>(
        builder: (context, appState, _) {
          return InkWell(
            onTap: () => _showModelPicker(context, appState),
            borderRadius: BorderRadius.circular(8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'New Conversation',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                ),
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      appState.selectedModel.name,
                      style: TextStyle(
                        fontSize: 13,
                        color: Colors.grey.shade600,
                      ),
                    ),
                    const SizedBox(width: 4),
                    Icon(
                      Icons.image_outlined,
                      size: 14,
                      color: Colors.grey.shade600,
                    ),
                    Icon(
                      Icons.arrow_drop_down,
                      size: 18,
                      color: Colors.grey.shade600,
                    ),
                  ],
                ),
              ],
            ),
          );
        },
      ),
      actions: [
        IconButton(
          icon: const Icon(Icons.tune_outlined),
          onPressed: () {},
        ),
      ],
    );
  }

  void _showModelPicker(BuildContext context, AppState appState) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) {
        final maxHeight = MediaQuery.of(ctx).size.height * 0.75;
        return SafeArea(
          child: ConstrainedBox(
            constraints: BoxConstraints(maxHeight: maxHeight),
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Container(
                      width: 40,
                      height: 4,
                      decoration: BoxDecoration(
                        color: Colors.grey.shade300,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),
                  const Text(
                    'Select Model',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 16),
                  Expanded(
                    child: ListView(
                      shrinkWrap: true,
                      children: appState.models.map((model) {
                        final isSelected = model.id == appState.selectedModelId;
                        return ListTile(
                          leading: Icon(
                            Icons.memory,
                            color: isSelected
                                ? Theme.of(context).colorScheme.primary
                                : Colors.grey,
                          ),
                          title: Text(
                            model.name,
                            style: TextStyle(
                              fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
                              color: isSelected
                                  ? Theme.of(context).colorScheme.primary
                                  : null,
                            ),
                          ),
                          subtitle: Text(model.size),
                          trailing: isSelected
                              ? Icon(Icons.check_circle,
                                  color: Theme.of(context).colorScheme.primary)
                              : null,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                          onTap: () {
                            appState.selectModel(model.id);
                            Navigator.pop(ctx);
                          },
                        );
                      }).toList(),
                    ),
                  ),
                  const SizedBox(height: 8),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildBody() {
    if (_currentIndex == 1 && _inChatView) {
      return const ChatScreen();
    }

    switch (_currentIndex) {
      case 0:
        return HomeScreen(onStartChat: _openChat);
      case 1:
        return ChatsScreen(onOpenChat: (conv) => _openChat(conv));
      case 2:
        return const ModelsScreen();
      case 3:
        return const SettingsScreen();
      default:
        return const SizedBox.shrink();
    }
  }

  Widget _buildBottomNav(BuildContext context) {
    return NavigationBar(
      selectedIndex: _currentIndex,
      onDestinationSelected: (index) {
        setState(() {
          _currentIndex = index;
          if (index != 1) _inChatView = false;
        });
      },
      backgroundColor: Colors.white,
      indicatorColor: Theme.of(context).colorScheme.primaryContainer,
      destinations: const [
        NavigationDestination(
          icon: Icon(Icons.home_outlined),
          selectedIcon: Icon(Icons.home),
          label: 'Home',
        ),
        NavigationDestination(
          icon: Icon(Icons.chat_bubble_outline),
          selectedIcon: Icon(Icons.chat_bubble),
          label: 'Chats',
        ),
        NavigationDestination(
          icon: Icon(Icons.memory_outlined),
          selectedIcon: Icon(Icons.memory),
          label: 'Models',
        ),
        NavigationDestination(
          icon: Icon(Icons.settings_outlined),
          selectedIcon: Icon(Icons.settings),
          label: 'Settings',
        ),
      ],
    );
  }
}
