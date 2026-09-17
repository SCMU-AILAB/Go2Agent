import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';

/// Section anchors of the single mission control page. The rail has no routes:
/// it scrolls the operator to the section that already exists on the page.
class ConsoleAnchors {
  ConsoleAnchors();

  final GlobalKey mission = GlobalKey(debugLabel: 'mission');
  final GlobalKey robot = GlobalKey(debugLabel: 'robot');
  final GlobalKey system = GlobalKey(debugLabel: 'system');
}

class ConsoleRail extends StatelessWidget {
  const ConsoleRail({
    super.key,
    required this.width,
    required this.anchors,
    required this.onNavigate,
  });

  final double width;
  final ConsoleAnchors anchors;
  final void Function(GlobalKey key) onNavigate;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: width,
      decoration: const BoxDecoration(
        color: ConsoleColors.bg1,
        border: Border(right: BorderSide(color: ConsoleColors.line)),
      ),
      child: Column(
        children: [
          const SizedBox(height: 18),
          Container(
            width: 30,
            height: 30,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: ConsoleColors.accent.withValues(alpha: .06),
              borderRadius: BorderRadius.circular(7),
              border: Border.all(
                color: ConsoleColors.accent.withValues(alpha: .3),
              ),
            ),
            child: const Text(
              'MC',
              style: TextStyle(
                color: ConsoleColors.accent,
                fontSize: 11,
                fontWeight: FontWeight.w800,
                letterSpacing: .5,
              ),
            ),
          ),
          const SizedBox(height: 26),
          _railItem(
            Icons.precision_manufacturing_outlined,
            'ROBOT',
            anchors.robot,
          ),
          const SizedBox(height: 8),
          _railItem(Icons.flag_outlined, 'MISSION', anchors.mission),
          const SizedBox(height: 8),
          _railItem(Icons.dns_outlined, 'SYSTEM', anchors.system),
          const Spacer(),
          const Text(
            'v0.1',
            style: TextStyle(
              color: ConsoleColors.faint,
              fontSize: 11,
              fontFamily: 'monospace',
              fontFamilyFallback: ['PingFang SC'],
            ),
          ),
          const SizedBox(height: 18),
        ],
      ),
    );
  }

  Widget _railItem(IconData icon, String label, GlobalKey target) {
    return Tooltip(
      message: label,
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: () => onNavigate(target),
          borderRadius: BorderRadius.circular(8),
          hoverColor: ConsoleColors.bg2,
          child: SizedBox(
            width: 56,
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 11),
              child: Column(
                children: [
                  Icon(icon, size: 19, color: ConsoleColors.muted),
                  const SizedBox(width: 0, height: 5),
                  Text(
                    label,
                    style: const TextStyle(
                      fontSize: 9,
                      fontWeight: FontWeight.w600,
                      letterSpacing: .6,
                      color: ConsoleColors.dim,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
