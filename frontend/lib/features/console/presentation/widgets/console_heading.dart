import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';
import 'console_widgets.dart';

class ConsoleHeading extends StatelessWidget {
  const ConsoleHeading({
    super.key,
    required this.controller,
    required this.width,
  });

  final ConsoleController controller;
  final double width;

  @override
  Widget build(BuildContext context) {
    final mobile = width <= 640;
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              hudLabel('AI ROBOTICS MISSION CONTROL'),
              const SizedBox(height: 6),
              Wrap(
                crossAxisAlignment: WrapCrossAlignment.center,
                spacing: mobile ? 8 : 12,
                runSpacing: 4,
                children: [
                  Text(
                    '机器人控制台',
                    style: TextStyle(
                      color: ConsoleColors.ink,
                      fontSize: mobile ? 18 : 22,
                      fontWeight: FontWeight.w600,
                      letterSpacing: -0.3,
                    ),
                  ),
                  tag(
                    controller.robotModelLabel,
                    foreground: ConsoleColors.accent,
                    border: ConsoleColors.accent.withValues(alpha: .3),
                    background: ConsoleColors.accent.withValues(alpha: .06),
                  ),
                ],
              ),
              const SizedBox(height: 4),
              Text(
                'TASK → AGENT → SKILL → ROBOT → VISION → RESULT',
                style: TextStyle(
                  fontSize: mobile ? 9 : 10,
                  fontFamily: 'monospace',
                  fontFamilyFallback: ['PingFang SC'],
                  letterSpacing: 0.6,
                  color: ConsoleColors.dim,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 10),
        Padding(
          padding: const EdgeInsets.only(bottom: 2),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              dot(
                controller.backend ? ConsoleColors.green : ConsoleColors.dim,
                glow: controller.backend,
              ),
              const SizedBox(width: 8),
              Text(
                controller.starting
                    ? 'STARTING'
                    : controller.backend
                    ? 'BACKEND ONLINE'
                    : 'BACKEND OFFLINE',
                style: TextStyle(
                  fontSize: mobile ? 10 : 11,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.8,
                  color: controller.backend
                      ? ConsoleColors.green
                      : ConsoleColors.muted,
                ),
              ),
              if (!mobile) ...[
                const SizedBox(width: 14),
                Container(width: 1, height: 12, color: ConsoleColors.line),
                const SizedBox(width: 14),
                Text(
                  'SESSION  ${controller.sessionId}',
                  style: const TextStyle(
                    fontSize: 11,
                    fontFamily: 'monospace',
                    fontFamilyFallback: ['PingFang SC'],
                    color: ConsoleColors.muted,
                  ),
                ),
              ],
            ],
          ),
        ),
      ],
    );
  }
}
