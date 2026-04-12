import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:picsprompt/main.dart';
import 'package:picsprompt/models/app_state.dart';

void main() {
  testWidgets('PicsPrompt app shell renders', (WidgetTester tester) async {
    await tester.pumpWidget(
      ChangeNotifierProvider(
        create: (_) => AppState(),
        child: const PicsPrompt(),
      ),
    );

    expect(find.text('Home'), findsWidgets);
    expect(find.text('PicsPrompt'), findsWidgets);
  });
}
