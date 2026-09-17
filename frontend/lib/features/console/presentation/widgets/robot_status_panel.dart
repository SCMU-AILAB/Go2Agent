import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';
import 'console_widgets.dart';

/// Collapsible robot identity block.
///
/// Collapsed by default: the operator's main job is issuing tasks, not reading
/// parameters. Only fields the backend actually reports are listed.
class RobotStatusPanel extends StatefulWidget {
  const RobotStatusPanel({super.key, required this.controller, this.anchorKey});

  final ConsoleController controller;
  final Key? anchorKey;

  @override
  State<RobotStatusPanel> createState() => _RobotStatusPanelState();
}

class _RobotStatusPanelState extends State<RobotStatusPanel> {
  bool expanded = false;

  void _toggle() => setState(() => expanded = !expanded);

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    final connected = controller.robotConnected;
    return Container(
      key: widget.anchorKey,
      clipBehavior: Clip.antiAlias,
      decoration: BoxDecoration(
        color: ConsoleColors.panel,
        border: Border.all(color: ConsoleColors.line),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          InkWell(
            onTap: _toggle,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
              child: Row(
                children: [
                  Icon(
                    Icons.precision_manufacturing_outlined,
                    size: 16,
                    color: ConsoleColors.muted,
                  ),
                  const SizedBox(width: 8),
                  const Expanded(
                    child: Text(
                      'ROBOT STATUS',
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        color: ConsoleColors.ink,
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 0.2,
                      ),
                    ),
                  ),
                  if (!expanded)
                    Flexible(
                      child: Text(
                        '${controller.robotModelLabel} · ${connected ? 'CONNECTED' : 'DISCONNECTED'} · ${controller.isHardware ? 'HW' : 'SIM'}',
                        overflow: TextOverflow.ellipsis,
                        textAlign: TextAlign.end,
                        style: const TextStyle(
                          color: ConsoleColors.dim,
                          fontSize: 10,
                          fontFamily: 'monospace',
                          fontFamilyFallback: ['PingFang SC'],
                        ),
                      ),
                    ),
                  const SizedBox(width: 8),
                  AnimatedRotation(
                    turns: expanded ? .5 : 0,
                    duration: const Duration(milliseconds: 180),
                    curve: Curves.easeOut,
                    child: const Icon(
                      Icons.expand_more_rounded,
                      size: 18,
                      color: ConsoleColors.dim,
                    ),
                  ),
                ],
              ),
            ),
          ),
          AnimatedCrossFade(
            firstChild: const SizedBox(width: double.infinity),
            secondChild: Container(
              width: double.infinity,
              padding: const EdgeInsets.fromLTRB(14, 2, 14, 14),
              decoration: const BoxDecoration(
                border: Border(top: BorderSide(color: ConsoleColors.lineSoft)),
              ),
              child: Column(
                children: [
                  telemetryRow(
                    'MODEL',
                    controller.robotModelLabel,
                    valueColor: ConsoleColors.accent,
                  ),
                  telemetryRow(
                    'CONNECTION',
                    connected ? 'CONNECTED' : 'DISCONNECTED',
                    valueColor: connected
                        ? ConsoleColors.green
                        : ConsoleColors.red,
                  ),
                  telemetryRow(
                    'MODE',
                    controller.isHardware ? 'HARDWARE' : 'SIMULATION',
                  ),
                  telemetryRow(
                    'ONBOARD TELEMETRY',
                    controller.telemetryUnavailable ? 'UNAVAILABLE' : 'N/A',
                    valueColor: controller.telemetryUnavailable
                        ? ConsoleColors.amber
                        : ConsoleColors.muted,
                  ),
                  telemetryRow(
                    'LATENCY',
                    controller.backend ? '${controller.latency} ms' : '—',
                  ),
                ],
              ),
            ),
            crossFadeState: expanded
                ? CrossFadeState.showSecond
                : CrossFadeState.showFirst,
            duration: const Duration(milliseconds: 200),
          ),
        ],
      ),
    );
  }
}
