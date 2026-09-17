import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';

/// Left rail: two top-level views only.
enum ConsoleView { vision, dialog }

class ConsoleRail extends StatelessWidget {
  const ConsoleRail({
    super.key,
    required this.width,
    required this.selected,
    required this.onSelect,
  });

  final double width;
  final ConsoleView selected;
  final void Function(ConsoleView view) onSelect;

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
          const SizedBox(height: 28),
          _railItem(
            icon: Icons.videocam_outlined,
            label: '视觉',
            selected: selected == ConsoleView.vision,
            onTap: () => onSelect(ConsoleView.vision),
          ),
          const SizedBox(height: 10),
          _railItem(
            icon: Icons.forum_outlined,
            label: '文字',
            selected: selected == ConsoleView.dialog,
            onTap: () => onSelect(ConsoleView.dialog),
          ),
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

  Widget _railItem({
    required IconData icon,
    required String label,
    required bool selected,
    required VoidCallback onTap,
  }) {
    final color = selected ? ConsoleColors.accent : ConsoleColors.muted;
    return Tooltip(
      message: label == '视觉' ? '视觉交互' : '文字交互',
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(8),
          hoverColor: ConsoleColors.bg2,
          child: Container(
            width: 56,
            padding: const EdgeInsets.symmetric(vertical: 11),
            decoration: BoxDecoration(
              color: selected
                  ? ConsoleColors.accent.withValues(alpha: .08)
                  : Colors.transparent,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
                color: selected
                    ? ConsoleColors.accent.withValues(alpha: .35)
                    : Colors.transparent,
              ),
            ),
            child: Column(
              children: [
                Icon(icon, size: 19, color: color),
                const SizedBox(height: 5),
                Text(
                  label,
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    letterSpacing: .4,
                    color: selected ? ConsoleColors.accent : ConsoleColors.dim,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
