import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';
import 'console_widgets.dart';

class ToolPanel extends StatelessWidget {
  const ToolPanel({super.key, required this.controller});

  final ConsoleController controller;

  @override
  Widget build(BuildContext context) {
    final child = controller.tools.isEmpty
        ? emptyState(
            Icons.handyman_outlined,
            '暂无工具调用',
            'Agent 调用 Skill 时会显示参数与结果',
          )
        : ListView.separated(
            padding: EdgeInsets.zero,
            itemCount: controller.tools.length,
            separatorBuilder: (context, index) => const SizedBox(height: 8),
            itemBuilder: (context, index) {
              final tool = controller.tools[index];
              return Material(
                color: ConsoleColors.field,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(6),
                  side: const BorderSide(color: ConsoleColors.lineSoft),
                ),
                child: Theme(
                  data: Theme.of(
                    context,
                  ).copyWith(dividerColor: Colors.transparent),
                  child: ExpansionTile(
                    tilePadding: const EdgeInsets.symmetric(horizontal: 9),
                    childrenPadding: EdgeInsets.zero,
                    dense: true,
                    iconColor: ConsoleColors.muted,
                    collapsedIconColor: ConsoleColors.dim,
                    title: Text(
                      tool.name,
                      style: const TextStyle(
                        color: ConsoleColors.accent,
                        fontSize: 12,
                        fontFamily: 'monospace',
                        fontFamilyFallback: ['PingFang SC'],
                      ),
                    ),
                    trailing: const Icon(
                      Icons.check_rounded,
                      size: 16,
                      color: ConsoleColors.green,
                    ),
                    children: [
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(9),
                        color: ConsoleColors.bg2,
                        child: Text(
                          tool.payload,
                          style: const TextStyle(
                            color: ConsoleColors.muted,
                            fontSize: 11,
                            height: 1.55,
                            fontFamily: 'monospace',
                            fontFamilyFallback: ['PingFang SC'],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              );
            },
          );
    return executionPanel(
      title: coloredTitle(
        Icons.handyman_outlined,
        'TOOL CALLS',
        Colors.transparent,
        ConsoleColors.amber,
      ),
      trailing: tag(
        '${controller.tools.length}',
        foreground: ConsoleColors.muted,
      ),
      child: child,
      footer: Row(
        children: [
          const Text('TOOLS'),
          const Spacer(),
          Text(
            controller.tools.isEmpty
                ? '尚未调用'
                : '${controller.tools.length} 次调用成功',
          ),
        ],
      ),
    );
  }
}
