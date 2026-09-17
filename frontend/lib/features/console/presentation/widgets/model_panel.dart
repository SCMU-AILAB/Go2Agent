import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';
import 'console_widgets.dart';

class ModelPanel extends StatelessWidget {
  const ModelPanel({super.key, required this.controller});

  final ConsoleController controller;

  @override
  Widget build(BuildContext context) {
    final content = controller.modelOutput.isEmpty
        ? emptyState(
            Icons.hourglass_empty_rounded,
            '等待任务指令',
            'Agent 分析与执行结果将在这里显示',
          )
        : SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (controller.currentTask.isNotEmpty) ...[
                  hudLabel('TASK'),
                  const SizedBox(height: 6),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: ConsoleColors.field,
                      borderRadius: BorderRadius.circular(5),
                      border: Border.all(color: ConsoleColors.lineSoft),
                    ),
                    child: Text(
                      controller.currentTask,
                      style: const TextStyle(
                        color: ConsoleColors.muted,
                        fontSize: 12,
                        height: 1.45,
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                ],
                hudLabel(
                  'AGENT · ${controller.isHardware ? 'HARDWARE' : 'SIMULATION'}',
                ),
                const SizedBox(height: 6),
                Text(
                  controller.modelOutput,
                  style: const TextStyle(
                    color: ConsoleColors.ink,
                    fontSize: 12,
                    height: 1.7,
                  ),
                ),
              ],
            ),
          );
    return executionPanel(
      title: coloredTitle(
        Icons.bolt_rounded,
        'AGENT OUTPUT',
        Colors.transparent,
        ConsoleColors.purple,
      ),
      trailing: statusTag(controller.modelStatus),
      child: content,
      footer: Row(
        children: [
          dot(ConsoleColors.purple),
          const SizedBox(width: 6),
          Expanded(
            child: Text(
              controller.cameraSource == 'local'
                  ? 'Video VLM / SkillRuntime'
                  : 'LangChain Agent / Ollama',
              overflow: TextOverflow.ellipsis,
            ),
          ),
          const SizedBox(width: 8),
          Text(
            '${(controller.modelDuration.inMilliseconds / 1000).toStringAsFixed(1)} s',
          ),
        ],
      ),
    );
  }
}
