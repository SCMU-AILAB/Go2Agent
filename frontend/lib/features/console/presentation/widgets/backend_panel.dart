import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';
import 'console_widgets.dart';

class BackendPanel extends StatelessWidget {
  const BackendPanel({super.key, required this.controller, this.anchorKey});

  final ConsoleController controller;
  final Key? anchorKey;

  @override
  Widget build(BuildContext context) {
    return Container(
      key: anchorKey,
      child: panel(
        header: sectionTitle(Icons.dns_outlined, 'SYSTEM'),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            dot(
              controller.backend ? ConsoleColors.green : ConsoleColors.dim,
              glow: controller.backend,
            ),
            const SizedBox(width: 6),
            Text(
              controller.starting
                  ? 'STARTING'
                  : controller.backend
                  ? 'ONLINE'
                  : 'OFFLINE',
              style: TextStyle(
                color: controller.backend
                    ? ConsoleColors.green
                    : ConsoleColors.dim,
                fontSize: 10,
                fontWeight: FontWeight.w700,
                letterSpacing: 1,
              ),
            ),
          ],
        ),
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(
                width: double.infinity,
                height: 40,
                child: OutlinedButton.icon(
                  onPressed: controller.starting
                      ? null
                      : controller.toggleBackend,
                  icon: Icon(
                    controller.backend
                        ? Icons.stop_rounded
                        : Icons.play_arrow_rounded,
                    size: 16,
                  ),
                  label: Text(
                    controller.starting
                        ? '正在启动…'
                        : controller.backend
                        ? '停止后端'
                        : '启动后端',
                    style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: controller.backend
                        ? ConsoleColors.muted
                        : ConsoleColors.accent,
                    backgroundColor: ConsoleColors.field,
                    side: BorderSide(
                      color: controller.backend
                          ? ConsoleColors.line
                          : ConsoleColors.accent.withValues(alpha: .4),
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(6),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              telemetryRow(
                'BACKEND',
                controller.starting
                    ? 'INIT'
                    : controller.backend
                    ? 'RUNNING'
                    : controller.connectedToApi
                    ? 'API ONLY'
                    : 'OFFLINE',
                valueColor: controller.backend
                    ? ConsoleColors.green
                    : ConsoleColors.muted,
              ),
              telemetryRow(
                'MODEL',
                controller.modelStatus,
                valueColor: controller.busy
                    ? ConsoleColors.accent
                    : ConsoleColors.ink,
              ),
              telemetryRow(
                'SKILL',
                controller.skillStatus,
                valueColor: controller.busy
                    ? ConsoleColors.accent
                    : ConsoleColors.muted,
              ),
              telemetryRow(
                'ROBOT',
                controller.robotConnected ? 'CONNECTED' : 'DISCONNECTED',
                valueColor: controller.robotConnected
                    ? ConsoleColors.green
                    : ConsoleColors.red,
              ),
              telemetryRow(
                'LATENCY',
                controller.backend ? '${controller.latency} ms' : '—',
              ),
              telemetryRow('CAMERA', controller.cameraStatus.toUpperCase()),
              telemetryRow('SESSION', controller.sessionId),
            ],
          ),
        ),
      ),
    );
  }
}
