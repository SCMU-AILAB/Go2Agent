import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';
import 'console_widgets.dart';

class SkillPanel extends StatelessWidget {
  const SkillPanel({super.key, required this.controller});

  final ConsoleController controller;

  @override
  Widget build(BuildContext context) {
    return executionPanel(
      title: coloredTitle(
        Icons.play_circle_outline_rounded,
        'SKILL PIPELINE',
        Colors.transparent,
        ConsoleColors.accent,
      ),
      childPadding: const EdgeInsets.all(14),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      hudLabel('CURRENT SKILL'),
                      const SizedBox(height: 4),
                      Text(
                        controller.skillName,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          color: ConsoleColors.ink,
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          fontFamily: 'monospace',
                          fontFamilyFallback: ['PingFang SC'],
                        ),
                      ),
                    ],
                  ),
                ),
                statusTag(controller.skillStatus),
              ],
            ),
            const SizedBox(height: 14),
            _step(0, '感知', '获取摄像头与场景信息'),
            _step(1, '规划', '解析指令并安排执行'),
            _step(2, '执行', '调用技能并反馈结果', last: true),
            const SizedBox(height: 4),
            ClipRRect(
              borderRadius: BorderRadius.circular(2),
              child: LinearProgressIndicator(
                value: controller.progress / 100,
                minHeight: 3,
                color: ConsoleColors.accent,
                backgroundColor: ConsoleColors.bg2,
              ),
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Text(
                    controller.progressText,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      color: ConsoleColors.dim,
                      fontSize: 11,
                    ),
                  ),
                ),
                Text(
                  '${controller.progress}%',
                  style: const TextStyle(
                    color: ConsoleColors.muted,
                    fontSize: 11,
                    fontFamily: 'monospace',
                    fontFamilyFallback: ['PingFang SC'],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _step(int index, String title, String caption, {bool last = false}) {
    final done = controller.activeStep > index;
    final active = controller.activeStep == index && controller.busy;
    final circleColor = active
        ? ConsoleColors.accent
        : done
        ? ConsoleColors.green.withValues(alpha: .15)
        : ConsoleColors.bg2;
    final borderColor = active
        ? ConsoleColors.accent
        : done
        ? ConsoleColors.green.withValues(alpha: .45)
        : ConsoleColors.line;
    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Column(
            children: [
              Container(
                width: 18,
                height: 18,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: circleColor,
                  shape: BoxShape.circle,
                  border: Border.all(color: borderColor),
                ),
                child: done
                    ? const Icon(
                        Icons.check,
                        size: 11,
                        color: ConsoleColors.green,
                      )
                    : Text(
                        '${index + 1}',
                        style: TextStyle(
                          fontSize: 9,
                          fontFamily: 'monospace',
                          fontFamilyFallback: const ['PingFang SC'],
                          color: active
                              ? ConsoleColors.bg0
                              : ConsoleColors.dim,
                        ),
                      ),
              ),
              if (!last)
                Expanded(
                  child: Container(
                    width: 1,
                    margin: const EdgeInsets.symmetric(vertical: 2),
                    color: ConsoleColors.line,
                  ),
                ),
            ],
          ),
          const SizedBox(width: 9),
          Expanded(
            child: Padding(
              padding: EdgeInsets.only(bottom: last ? 8 : 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      color: active
                          ? ConsoleColors.accent
                          : ConsoleColors.ink,
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const SizedBox(height: 1),
                  Text(
                    caption,
                    style: const TextStyle(
                      color: ConsoleColors.dim,
                      fontSize: 11,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
