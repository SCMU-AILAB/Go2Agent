import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';
import 'console_widgets.dart';
import 'estop_button.dart';

class ConsoleHeader extends StatelessWidget {
  const ConsoleHeader({
    super.key,
    required this.controller,
    required this.width,
  });

  final ConsoleController controller;
  final double width;

  @override
  Widget build(BuildContext context) {
    final mobile = width <= 640;
    final online = controller.robotConnected;
    final model = controller.robotModelLabel;
    return Container(
      height: mobile ? 58 : 64,
      padding: EdgeInsets.symmetric(horizontal: mobile ? 12 : 24),
      decoration: const BoxDecoration(
        color: ConsoleColors.bg1,
        border: Border(bottom: BorderSide(color: ConsoleColors.line)),
      ),
      child: Row(
        children: [
          Text(
            model,
            style: TextStyle(
              color: ConsoleColors.ink,
              fontSize: mobile ? 13 : 16,
              fontWeight: FontWeight.w700,
              letterSpacing: 1.4,
            ),
          ),
          const SizedBox(width: 8),
          tag(
            mobile
                ? (controller.isHardware ? 'HW' : 'SIM')
                : (controller.isHardware
                      ? 'HARDWARE · 真机模式'
                      : 'SIMULATION · 模拟模式'),
            foreground: controller.isHardware
                ? ConsoleColors.accent
                : ConsoleColors.muted,
            background: controller.isHardware
                ? ConsoleColors.accent.withValues(alpha: .08)
                : ConsoleColors.bg2,
            border: controller.isHardware
                ? ConsoleColors.accent.withValues(alpha: .35)
                : ConsoleColors.line,
          ),
          if (!mobile) const SizedBox(width: 10),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              dot(
                online ? ConsoleColors.green : ConsoleColors.dim,
                glow: online,
              ),
              const SizedBox(width: 6),
              Text(
                online ? 'ONLINE' : 'OFFLINE',
                style: TextStyle(
                  color: online ? ConsoleColors.green : ConsoleColors.dim,
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.8,
                ),
              ),
            ],
          ),
          const Spacer(),
          EmergencyStopButton(controller: controller),
          if (!mobile) const SizedBox(width: 12),
          if (!mobile) ...[
            _metric('CAMERA', _cameraLabel()),
            const SizedBox(width: 14),
            _metric(
              'LATENCY',
              controller.backend ? '${controller.latency} ms' : '—',
              mono: true,
            ),
            const SizedBox(width: 14),
            _metric('TASKS', '${controller.taskCount}', mono: true),
            const SizedBox(width: 14),
            Text(
              controller.formattedTime(),
              style: const TextStyle(
                color: ConsoleColors.muted,
                fontFamily: 'monospace',
                fontFamilyFallback: ['PingFang SC'],
                fontSize: 12,
              ),
            ),
          ] else
            Flexible(
              child: Text(
                controller.formattedTime(),
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  color: ConsoleColors.muted,
                  fontFamily: 'monospace',
                  fontFamilyFallback: ['PingFang SC'],
                  fontSize: 11,
                ),
              ),
            ),
        ],
      ),
    );
  }

  String _cameraLabel() {
    switch (controller.cameraStatus) {
      case 'ready':
        return 'OK';
      case 'error':
        return 'ERR';
      case 'starting':
        return '…';
      default:
        return 'IDLE';
    }
  }

  Widget _metric(String label, String value, {bool mono = false}) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label,
          style: const TextStyle(
            color: ConsoleColors.dim,
            fontSize: 9,
            fontWeight: FontWeight.w600,
            letterSpacing: 1.2,
          ),
        ),
        const SizedBox(width: 6),
        Text(
          value,
          style: TextStyle(
            color: ConsoleColors.ink,
            fontSize: 12,
            fontWeight: FontWeight.w600,
            fontFamily: mono ? 'monospace' : null,
          ),
        ),
      ],
    );
  }
}
