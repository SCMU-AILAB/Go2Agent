import 'package:flutter/services.dart';

import '../../models/console_log.dart';
import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';
import 'console_widgets.dart';

class LogPanel extends StatelessWidget {
  const LogPanel({super.key, required this.controller, required this.width});

  final ConsoleController controller;
  final double width;

  @override
  Widget build(BuildContext context) {
    final mobile = width <= 640;
    return panel(
      headerHeight: null,
      header: Wrap(
        crossAxisAlignment: WrapCrossAlignment.center,
        spacing: 12,
        children: [
          sectionTitle(Icons.terminal_rounded, 'SYSTEM LOG'),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              dot(ConsoleColors.accent, glow: true),
              const SizedBox(width: 6),
              const Text(
                'LIVE',
                style: TextStyle(
                  color: ConsoleColors.muted,
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 1,
                ),
              ),
            ],
          ),
          Text(
            '${controller.logs.length} 条',
            style: const TextStyle(color: ConsoleColors.dim, fontSize: 11),
          ),
        ],
      ),
      trailing: Wrap(
        alignment: WrapAlignment.end,
        crossAxisAlignment: WrapCrossAlignment.center,
        spacing: 5,
        runSpacing: 6,
        children: [
          SizedBox(
            width: mobile ? 112 : 122,
            height: 32,
            child: DropdownButtonFormField<String>(
              initialValue: controller.logLevel,
              isDense: true,
              dropdownColor: ConsoleColors.panelElevated,
              style: const TextStyle(
                color: ConsoleColors.muted,
                fontSize: 11,
                fontFamily: 'PingFang SC',
              ),
              iconEnabledColor: ConsoleColors.muted,
              decoration: const InputDecoration(
                contentPadding: EdgeInsets.symmetric(
                  horizontal: 9,
                  vertical: 6,
                ),
              ),
              items: const [
                DropdownMenuItem(value: 'ALL', child: Text('全部等级')),
                DropdownMenuItem(value: 'INFO', child: Text('INFO')),
                DropdownMenuItem(value: 'WARN', child: Text('WARN')),
                DropdownMenuItem(value: 'ERROR', child: Text('ERROR')),
                DropdownMenuItem(value: 'DEBUG', child: Text('DEBUG')),
              ],
              onChanged: (value) => controller.setLogLevel(value ?? 'ALL'),
            ),
          ),
          SizedBox(
            width: mobile ? 118 : 145,
            height: 32,
            child: TextField(
              controller: controller.searchController,
              style: const TextStyle(fontSize: 11, color: ConsoleColors.ink),
              cursorColor: ConsoleColors.accent,
              decoration: const InputDecoration(
                hintText: '搜索日志…',
                contentPadding: EdgeInsets.symmetric(
                  horizontal: 9,
                  vertical: 6,
                ),
              ),
            ),
          ),
          _logAction('清空', controller.clearLogs),
          IconButton(
            tooltip: '复制全部日志',
            onPressed: () {
              final text = controller.logs
                  .map(
                    (e) =>
                        '[${e.time}] [${e.level}] [${e.source}] ${e.message}',
                  )
                  .join('\n');
              Clipboard.setData(ClipboardData(text: text));
              controller.onMessage('全部日志已复制到剪贴板');
            },
            icon: const Icon(
              Icons.copy_all_outlined,
              size: 17,
              color: ConsoleColors.muted,
            ),
          ),
        ],
      ),
      child: Column(
        children: [
          Container(
            height: mobile ? 210 : 180,
            color: ConsoleColors.bg2,
            child: controller.visibleLogs.isEmpty
                ? const Center(
                    child: Text(
                      '暂无匹配的日志',
                      style: TextStyle(color: ConsoleColors.dim, fontSize: 12),
                    ),
                  )
                : ListView.builder(
                    controller: controller.logScrollController,
                    padding: const EdgeInsets.symmetric(
                      horizontal: 14,
                      vertical: 10,
                    ),
                    itemCount: controller.visibleLogs.length,
                    itemBuilder: (context, index) =>
                        _logRow(controller.visibleLogs[index], mobile),
                  ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
            decoration: const BoxDecoration(
              border: Border(top: BorderSide(color: ConsoleColors.lineSoft)),
            ),
            child: Row(
              children: [
                const Text(
                  '自动滚动至最新日志',
                  style: TextStyle(color: ConsoleColors.faint, fontSize: 10),
                ),
                const Spacer(),
                if (!mobile)
                  Text(
                    '${controller.robotModelLabel} CONSOLE / ${controller.isHardware ? 'HARDWARE' : 'SIMULATION'}',
                    style: const TextStyle(
                      color: ConsoleColors.faint,
                      fontSize: 10,
                      fontFamily: 'monospace',
                      fontFamilyFallback: ['PingFang SC'],
                      letterSpacing: .5,
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _logRow(ConsoleLog entry, bool mobile) {
    final levelColor = switch (entry.level) {
      'INFO' => ConsoleColors.accent,
      'WARN' => ConsoleColors.amber,
      'ERROR' => ConsoleColors.red,
      _ => ConsoleColors.purple,
    };
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: mobile ? 72 : 88,
            child: Text(
              entry.time,
              style: const TextStyle(
                color: ConsoleColors.faint,
                fontSize: 11,
                fontFamily: 'monospace',
                fontFamilyFallback: ['PingFang SC'],
              ),
            ),
          ),
          SizedBox(
            width: mobile ? 48 : 54,
            child: Text(
              entry.level,
              style: TextStyle(
                color: levelColor,
                fontSize: 10,
                fontWeight: FontWeight.w700,
                fontFamily: 'monospace',
                fontFamilyFallback: ['PingFang SC'],
              ),
            ),
          ),
          if (!mobile)
            SizedBox(
              width: 92,
              child: Text(
                entry.source,
                style: const TextStyle(
                  color: ConsoleColors.muted,
                  fontSize: 11,
                  fontFamily: 'monospace',
                  fontFamilyFallback: ['PingFang SC'],
                ),
              ),
            ),
          Expanded(
            child: Text(
              entry.message,
              style: const TextStyle(
                color: ConsoleColors.ink,
                fontSize: 11,
                fontFamily: 'monospace',
                fontFamilyFallback: ['PingFang SC'],
                height: 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _logAction(String label, VoidCallback action) {
    return TextButton(
      onPressed: action,
      style: TextButton.styleFrom(
        foregroundColor: ConsoleColors.muted,
        padding: const EdgeInsets.symmetric(horizontal: 6),
        minimumSize: const Size(0, 30),
      ),
      child: Text(label, style: const TextStyle(fontSize: 11)),
    );
  }
}
