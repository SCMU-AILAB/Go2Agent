import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';

/// Always-visible operator e-stop in the top bar.
class EmergencyStopButton extends StatelessWidget {
  const EmergencyStopButton({super.key, required this.controller});

  final ConsoleController controller;

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: '急停：取消任务并下发机器人 stop',
      child: FilledButton.icon(
        onPressed: controller.emergencyStop,
        icon: const Icon(Icons.stop_circle_outlined, size: 16),
        label: const Text('急停'),
        style: FilledButton.styleFrom(
          backgroundColor: ConsoleColors.red,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          minimumSize: const Size(0, 34),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(6),
          ),
          textStyle: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
    );
  }
}
