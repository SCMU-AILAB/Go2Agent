import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';

Widget panel({
  Widget? header,
  Widget? trailing,
  required Widget child,
  bool paddingHeader = true,
  double? headerHeight = 48,
}) {
  return Container(
    clipBehavior: Clip.antiAlias,
    decoration: BoxDecoration(
      color: ConsoleColors.panel,
      border: Border.all(color: ConsoleColors.line),
      borderRadius: BorderRadius.circular(10),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (paddingHeader && header != null)
          Container(
            constraints: BoxConstraints(minHeight: headerHeight ?? 48),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: const BoxDecoration(
              border: Border(bottom: BorderSide(color: ConsoleColors.lineSoft)),
            ),
            child: Row(
              children: [
                Expanded(child: header),
                if (trailing != null) ...[
                  const SizedBox(width: 10),
                  Flexible(child: trailing),
                ],
              ],
            ),
          ),
        child,
      ],
    ),
  );
}

Widget executionPanel({
  required Widget title,
  Widget? trailing,
  required Widget child,
  Widget? footer,
  EdgeInsets childPadding = const EdgeInsets.all(14),
  double height = 340,
}) {
  return SizedBox(
    height: height,
    child: panel(
      header: title,
      trailing: trailing,
      child: Expanded(
        child: Column(
          children: [
            Expanded(
              child: Padding(padding: childPadding, child: child),
            ),
            if (footer != null)
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 14,
                  vertical: 8,
                ),
                decoration: const BoxDecoration(
                  border: Border(
                    top: BorderSide(color: ConsoleColors.lineSoft),
                  ),
                ),
                child: DefaultTextStyle(
                  style: const TextStyle(
                    color: ConsoleColors.dim,
                    fontSize: 10,
                    fontFamily: 'monospace',
                    fontFamilyFallback: ['PingFang SC'],
                  ),
                  child: footer,
                ),
              ),
          ],
        ),
      ),
    ),
  );
}

Widget sectionTitle(IconData icon, String title) {
  return Row(
    children: [
      Icon(icon, size: 16, color: ConsoleColors.muted),
      const SizedBox(width: 8),
      Expanded(
        child: Text(
          title,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(
            color: ConsoleColors.ink,
            fontSize: 13,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.2,
          ),
        ),
      ),
    ],
  );
}

Widget hudLabel(String text) {
  return Text(
    text,
    style: const TextStyle(
      color: ConsoleColors.dim,
      fontSize: 10,
      fontWeight: FontWeight.w600,
      letterSpacing: 1.4,
    ),
  );
}

Widget coloredTitle(
  IconData icon,
  String title,
  Color background,
  Color foreground,
) {
  return Row(
    children: [
      Icon(icon, color: foreground, size: 16),
      const SizedBox(width: 8),
      Expanded(
        child: Text(
          title,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(
            color: ConsoleColors.ink,
            fontSize: 13,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    ],
  );
}

Widget tag(
  String text, {
  Color foreground = ConsoleColors.muted,
  Color? background,
  Color? border,
}) {
  return Container(
    padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
    decoration: BoxDecoration(
      color: background ?? ConsoleColors.bg2,
      border: Border.all(color: border ?? ConsoleColors.line),
      borderRadius: BorderRadius.circular(4),
    ),
    child: Text(
      text,
      style: TextStyle(
        color: foreground,
        fontSize: 10,
        fontWeight: FontWeight.w600,
        letterSpacing: text.contains('DEMO') || text.contains('HARDWARE')
            ? 0.6
            : 0.2,
      ),
    ),
  );
}

Widget statusTag(String status) {
  if (status == 'RUNNING' || status == '生成中' || status == 'BUSY') {
    return tag(
      status,
      foreground: ConsoleColors.accent,
      border: ConsoleColors.accent.withValues(alpha: .35),
      background: ConsoleColors.accent.withValues(alpha: .08),
    );
  }
  if (status == 'DONE' || status == '已完成') {
    return tag(
      status,
      foreground: ConsoleColors.green,
      border: ConsoleColors.green.withValues(alpha: .35),
      background: ConsoleColors.green.withValues(alpha: .08),
    );
  }
  if (status == 'STOPPED' || status == '已停止') {
    return tag(
      status,
      foreground: ConsoleColors.amber,
      border: ConsoleColors.amber.withValues(alpha: .35),
      background: ConsoleColors.amber.withValues(alpha: .08),
    );
  }
  return tag(status);
}

Widget dot(Color color, {bool glow = false}) {
  return Container(
    width: 6,
    height: 6,
    decoration: BoxDecoration(
      color: color,
      shape: BoxShape.circle,
      boxShadow: glow
          ? [BoxShadow(color: color.withValues(alpha: .35), blurRadius: 6)]
          : null,
    ),
  );
}

Widget darkChip(String text) {
  return Container(
    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
    decoration: BoxDecoration(
      color: const Color(0xE60A0C10),
      border: Border.all(color: ConsoleColors.line.withValues(alpha: .9)),
      borderRadius: BorderRadius.circular(4),
    ),
    child: Text(
      text,
      overflow: TextOverflow.ellipsis,
      style: const TextStyle(
        color: ConsoleColors.muted,
        fontSize: 10,
        fontFamily: 'monospace',
        fontFamilyFallback: ['PingFang SC'],
        letterSpacing: .4,
      ),
    ),
  );
}

Widget glassButton(IconData icon, String tooltip, VoidCallback action) {
  return Tooltip(
    message: tooltip,
    child: InkWell(
      onTap: action,
      borderRadius: BorderRadius.circular(5),
      child: Container(
        width: 32,
        height: 30,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: const Color(0xCC0C0E12),
          border: Border.all(color: ConsoleColors.line),
          borderRadius: BorderRadius.circular(5),
        ),
        child: Icon(icon, size: 15, color: ConsoleColors.muted),
      ),
    ),
  );
}

Widget emptyState(Object icon, String title, String caption) {
  return Center(
    child: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        icon is IconData
            ? Icon(icon, color: ConsoleColors.faint, size: 24)
            : Text(
                icon.toString(),
                style: const TextStyle(
                  color: ConsoleColors.faint,
                  fontSize: 28,
                ),
              ),
        const SizedBox(height: 10),
        Text(
          title,
          textAlign: TextAlign.center,
          style: const TextStyle(color: ConsoleColors.muted, fontSize: 12),
        ),
        const SizedBox(height: 6),
        Text(
          caption,
          textAlign: TextAlign.center,
          style: const TextStyle(
            color: ConsoleColors.dim,
            fontSize: 11,
            height: 1.5,
          ),
        ),
      ],
    ),
  );
}

Widget telemetryRow(String label, String value, {Color? valueColor}) {
  return Padding(
    padding: const EdgeInsets.symmetric(vertical: 5),
    child: Row(
      children: [
        Expanded(
          child: Text(
            label,
            style: const TextStyle(color: ConsoleColors.dim, fontSize: 11),
          ),
        ),
        Text(
          value,
          style: TextStyle(
            color: valueColor ?? ConsoleColors.ink,
            fontSize: 11,
            fontWeight: FontWeight.w600,
            fontFamily: 'monospace',
            fontFamilyFallback: ['PingFang SC'],
          ),
        ),
      ],
    ),
  );
}
