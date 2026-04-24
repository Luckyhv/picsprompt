import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/app_state.dart';
import '../models/chat_message.dart';

class ModelsScreen extends StatelessWidget {
  const ModelsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<AppState>(
      builder: (context, appState, _) {
        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            ...appState.models.map((model) {
              final isSelected = model.id == appState.selectedModelId;
              return Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child:
                    _buildModelCard(context, appState, model, isSelected),
              );
            }),
          ],
        );
      },
    );
  }

  Widget _buildModelCard(
    BuildContext context,
    AppState appState,
    AIModel model,
    bool isSelected,
  ) {
    final colorScheme = Theme.of(context).colorScheme;
    final onnxReady = appState.isModelReady(model.id);
    final canRun = onnxReady;

    return Opacity(
      opacity: canRun ? 1.0 : 0.5,
      child: InkWell(
        onTap: canRun
            ? () {
                appState.selectModel(model.id);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text('Switched to ${model.name}'),
                    duration: const Duration(seconds: 1),
                    behavior: SnackBarBehavior.floating,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                );
              }
            : null,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            border: Border.all(
              color: isSelected ? colorScheme.primary : Colors.grey.shade200,
              width: isSelected ? 2 : 1,
            ),
            borderRadius: BorderRadius.circular(16),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    width: 52,
                    height: 52,
                    decoration: BoxDecoration(
                      color: isSelected
                          ? colorScheme.primaryContainer
                          : Colors.grey.shade100,
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: Icon(
                      Icons.auto_awesome,
                      color: isSelected
                          ? colorScheme.primary
                          : Colors.grey.shade600,
                    ),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          model.name,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            color: isSelected ? colorScheme.primary : null,
                          ),
                        ),
                        const SizedBox(height: 3),
                        Text(
                          model.description,
                          maxLines: 5,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            color: Colors.grey.shade600,
                            fontSize: 13,
                          ),
                        ),
                      ],
                    ),
                  ),
                  if (isSelected)
                    Icon(Icons.check_circle, color: colorScheme.primary)
                  else if (!canRun)
                    Icon(Icons.warning_amber_rounded,
                        color: Colors.orange.shade400)
                  else
                    Icon(Icons.radio_button_unchecked,
                        color: Colors.grey.shade400),
                ],
              ),
              if (!canRun) ...[
                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.orange.shade50,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(Icons.info_outline,
                          size: 14, color: Colors.orange.shade700),
                      const SizedBox(width: 6),
                      Expanded(
                        child: Text(
                          switch (model.id) {
                            'butterflies' =>
                              'No Butterflies ONNX on device. Run picsprompt-models/scripts/push_butterflies_to_android.sh, then restart app.',
                            'animegan' =>
                              'No AnimeGAN ONNX on device. Run picsprompt-models/scripts/push_animegan_to_android.sh, then restart app.',
                            _ =>
                              'No DreamShaper ONNX on device. Run picsprompt-models/scripts/push_dreamshaper_to_android.sh, then restart app.',
                          },
                          softWrap: true,
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.orange.shade700,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

}
