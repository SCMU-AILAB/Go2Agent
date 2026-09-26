import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';

class ConsoleFooter extends StatelessWidget {
  const ConsoleFooter({
    super.key,
    required this.controller,
    required this.width,
  });

  final ConsoleController controller;
  final double width;

  @override
  Widget build(BuildContext context) {
    final mobile = width <= 640;
    final text = Text(
      _statusText,
      style: const TextStyle(
        color: ConsoleColors.faint,
        fontSize: 10,
        fontFamily: 'monospace',
        fontFamilyFallback: ['PingFang SC'],
      ),
    );
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16),
      child: mobile
          ? Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'AI Robotics Mission Control',
                  style: TextStyle(color: ConsoleColors.faint, fontSize: 10),
                ),
                const SizedBox(height: 4),
                text,
              ],
            )
          : Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'AI Robotics Mission Control · Go2',
                  style: TextStyle(color: ConsoleColors.faint, fontSize: 10),
                ),
                text,
              ],
            ),
    );
  }

  String get _statusText {
    final robot = controller.robotModelLabel;
    return controller.isHardware
        ? 'FastAPI → SkillRuntime → Unitree $robot'
        : 'FastAPI → SkillRuntime → SimulatedRobotAdapter';
  }
}
