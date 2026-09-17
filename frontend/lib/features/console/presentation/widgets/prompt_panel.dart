import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';
import 'console_widgets.dart';

class PromptPanel extends StatelessWidget {
  const PromptPanel({super.key, required this.controller});

  final ConsoleController controller;

  @override
  Widget build(BuildContext context) {
    return panel(
      header: sectionTitle(Icons.tune_rounded, 'SYSTEM PROMPT'),
      trailing: tag(
        controller.promptSaved ? '已保存' : '未保存',
        foreground: controller.promptSaved
            ? ConsoleColors.muted
            : ConsoleColors.amber,
        border: controller.promptSaved
            ? ConsoleColors.line
            : ConsoleColors.amber.withValues(alpha: .4),
        background: controller.promptSaved
            ? ConsoleColors.bg2
            : ConsoleColors.amber.withValues(alpha: .08),
      ),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              controller.cameraSource == 'local'
                  ? '视觉任务偏好（不能覆盖手势与安全约束）'
                  : '定义机器人的行为与约束',
              style: const TextStyle(color: ConsoleColors.dim, fontSize: 11),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: controller.systemPromptController,
              minLines: 4,
              maxLines: 6,
              cursorColor: ConsoleColors.accent,
              style: const TextStyle(
                color: ConsoleColors.ink,
                fontSize: 12,
                height: 1.55,
              ),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Text(
                  '${controller.systemPromptController.text.length} 字',
                  style: const TextStyle(
                    color: ConsoleColors.faint,
                    fontSize: 11,
                  ),
                ),
                const Spacer(),
                TextButton(
                  onPressed: controller.savePrompt,
                  style: TextButton.styleFrom(
                    foregroundColor: ConsoleColors.accent,
                    padding: EdgeInsets.zero,
                    minimumSize: const Size(0, 28),
                  ),
                  child: const Text('保存配置  ↗', style: TextStyle(fontSize: 12)),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
