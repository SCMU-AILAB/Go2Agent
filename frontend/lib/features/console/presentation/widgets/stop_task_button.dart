import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';

class StopTaskButton extends StatelessWidget {
  const StopTaskButton({super.key, required this.controller});

  final ConsoleController controller;

  @override
  Widget build(BuildContext context) {
    final enabled = controller.busy;
    return SizedBox(
      width: double.infinity,
      height: 42,
      child: OutlinedButton(
        onPressed: enabled ? controller.cancelTask : null,
        style: OutlinedButton.styleFrom(
          foregroundColor: ConsoleColors.red,
          backgroundColor: enabled
              ? ConsoleColors.red.withValues(alpha: .08)
              : ConsoleColors.bg2,
          disabledForegroundColor: ConsoleColors.red.withValues(alpha: .35),
          side: BorderSide(
            color: enabled
                ? ConsoleColors.red.withValues(alpha: .55)
                : ConsoleColors.line,
          ),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
        ),
        child: Row(
          children: [
            const Icon(Icons.stop_rounded, size: 16),
            const Expanded(
              child: Center(
                child: Text(
                  '停止执行',
                  style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
                ),
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
              decoration: BoxDecoration(
                border: Border.all(
                  color: ConsoleColors.red.withValues(alpha: .4),
                ),
                borderRadius: BorderRadius.circular(3),
              ),
              child: const Text(
                'ESC',
                style: TextStyle(
                  fontFamily: 'monospace',
                  fontFamilyFallback: ['PingFang SC'],
                  fontSize: 10,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
