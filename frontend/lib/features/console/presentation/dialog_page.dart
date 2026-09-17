import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../controllers/console_controller.dart';
import 'widgets/console_header.dart';
import 'widgets/console_footer.dart';
import 'widgets/console_widgets.dart';
import 'widgets/log_panel.dart';

/// Standalone dialog / text-interaction page.
///
/// Chat history is accumulated in the controller from console snapshots
/// (no backend changes). Status mirrors text-task progress on the mission page.
class DialogPage extends StatefulWidget {
  const DialogPage({super.key, required this.controller});

  final ConsoleController controller;

  @override
  State<DialogPage> createState() => _DialogPageState();
}

class _DialogPageState extends State<DialogPage> {
  final _quickInput = TextEditingController();
  final _scroll = ScrollController();
  bool _sessionActive = false;

  ConsoleController get controller => widget.controller;

  @override
  void dispose() {
    _quickInput.dispose();
    _scroll.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scroll.hasClients) return;
      _scroll.animateTo(
        _scroll.position.maxScrollExtent,
        duration: const Duration(milliseconds: 200),
        curve: Curves.easeOut,
      );
    });
  }

  void _startSession() {
    if (!controller.backend || controller.busy) return;
    controller.setTaskMode('text');
    setState(() => _sessionActive = true);
    controller.onMessage('文字交互已开始，可输入指令发送');
  }

  void _stopSession() {
    if (controller.busy) {
      controller.cancelTask('停止文字交互');
    }
    setState(() => _sessionActive = false);
  }

  Future<void> _send() async {
    final text = _quickInput.text.trim();
    if (text.isEmpty || controller.busy) return;
    if (!_sessionActive) {
      _startSession();
    }
    controller.setTaskMode('text');
    controller.taskController.text = text;
    await controller.submitTask();
    if (mounted) {
      _quickInput.clear();
      _scrollToBottom();
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: controller,
      builder: (context, _) {
        WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());
        final width = MediaQuery.sizeOf(context).width;
        final wide = width > 900;
        return Column(
          children: [
            ConsoleHeader(controller: controller, width: width),
            Expanded(
              child: SingleChildScrollView(
                padding: EdgeInsets.fromLTRB(
                  width <= 640 ? 12 : 24,
                  width <= 640 ? 14 : 20,
                  width <= 640 ? 12 : 24,
                  0,
                ),
                child: Center(
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 1680),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        _PageTitle(controller: controller),
                        const SizedBox(height: 14),
                        if (wide)
                          Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Expanded(
                                child: _MainColumn(
                                  controller: controller,
                                  input: _quickInput,
                                  scroll: _scroll,
                                  onSend: _send,
                                  sessionActive: _sessionActive,
                                  onStart: _startSession,
                                  onStop: _stopSession,
                                ),
                              ),
                              const SizedBox(width: 16),
                              SizedBox(
                                width: 310,
                                child: _SideColumn(
                                  controller: controller,
                                  sessionActive: _sessionActive,
                                  onStart: _startSession,
                                  onStop: _stopSession,
                                ),
                              ),
                            ],
                          )
                        else ...[
                          _MainColumn(
                            controller: controller,
                            input: _quickInput,
                            scroll: _scroll,
                            onSend: _send,
                            sessionActive: _sessionActive,
                            onStart: _startSession,
                            onStop: _stopSession,
                          ),
                          const SizedBox(height: 12),
                          _SideColumn(
                            controller: controller,
                            sessionActive: _sessionActive,
                            onStart: _startSession,
                            onStop: _stopSession,
                          ),
                        ],
                        const SizedBox(height: 16),
                        LogPanel(controller: controller, width: width),
                        ConsoleFooter(controller: controller, width: width),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}

class _PageTitle extends StatelessWidget {
  const _PageTitle({required this.controller});

  final ConsoleController controller;

  @override
  Widget build(BuildContext context) {
    final c = controller;
    return Row(
      children: [
        Icon(Icons.forum_outlined, color: ConsoleColors.accent, size: 20),
        const SizedBox(width: 10),
        const Expanded(
          child: Text(
            '文字交互',
            style: TextStyle(
              color: ConsoleColors.ink,
              fontSize: 18,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(
            color: ConsoleColors.field,
            borderRadius: BorderRadius.circular(999),
            border: Border.all(color: ConsoleColors.line),
          ),
          child: Text(
            '${c.robotModelLabel} · ${c.isHardware ? '真机' : '模拟'} · '
            '${c.backend ? 'API 已连接' : 'API 未连接'}',
            style: const TextStyle(color: ConsoleColors.muted, fontSize: 11),
          ),
        ),
      ],
    );
  }
}

class _MainColumn extends StatelessWidget {
  const _MainColumn({
    required this.controller,
    required this.input,
    required this.scroll,
    required this.onSend,
    required this.sessionActive,
    required this.onStart,
    required this.onStop,
  });

  final ConsoleController controller;
  final TextEditingController input;
  final ScrollController scroll;
  final VoidCallback onSend;
  final bool sessionActive;
  final VoidCallback onStart;
  final VoidCallback onStop;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SizedBox(
          height: 420,
          child: _ChatBody(controller: controller, scroll: scroll),
        ),
        const SizedBox(height: 12),
        _Composer(
          controller: controller,
          input: input,
          onSend: onSend,
          sessionActive: sessionActive,
          onStart: onStart,
          onStop: onStop,
        ),
      ],
    );
  }
}

class _SideColumn extends StatelessWidget {
  const _SideColumn({
    required this.controller,
    required this.sessionActive,
    required this.onStart,
    required this.onStop,
  });

  final ConsoleController controller;
  final bool sessionActive;
  final VoidCallback onStart;
  final VoidCallback onStop;

  @override
  Widget build(BuildContext context) {
    final c = controller;
    final running = c.busy || sessionActive;
    final canStart = c.backend && !c.busy;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _StatusCard(controller: controller),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          height: 42,
          child: running
              ? OutlinedButton.icon(
                  onPressed: onStop,
                  icon: const Icon(Icons.stop_rounded, size: 18),
                  label: const Text('停止执行'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: ConsoleColors.red,
                    backgroundColor: ConsoleColors.red.withValues(alpha: .08),
                    side: BorderSide(
                      color: ConsoleColors.red.withValues(alpha: .55),
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(6),
                    ),
                  ),
                )
              : FilledButton.icon(
                  onPressed: canStart ? onStart : null,
                  iconAlignment: IconAlignment.end,
                  icon: const Icon(Icons.play_arrow_rounded, size: 18),
                  label: const Text('开始文本对话'),
                  style: FilledButton.styleFrom(
                    backgroundColor: ConsoleColors.accent,
                    foregroundColor: ConsoleColors.bg0,
                    disabledBackgroundColor: ConsoleColors.bg2,
                    disabledForegroundColor: ConsoleColors.faint,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(6),
                    ),
                    textStyle: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
        ),
      ],
    );
  }
}

class _StatusCard extends StatelessWidget {
  const _StatusCard({required this.controller});

  final ConsoleController controller;

  @override
  Widget build(BuildContext context) {
    final c = controller;
    Color statusColor = ConsoleColors.muted;
    String statusLabel = c.modelStatus;
    if (c.busy) {
      statusColor = ConsoleColors.accent;
      statusLabel = '处理中 · ${c.progressText}';
    } else if (c.skillStatus == 'FAILED') {
      statusColor = ConsoleColors.red;
      statusLabel = '上次失败';
    } else if (c.taskCount > 0) {
      statusColor = ConsoleColors.green;
      statusLabel = '待命 · 已完成 ${c.taskCount} 次任务';
    }

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: ConsoleColors.panel,
        border: Border.all(color: ConsoleColors.line),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.forum_outlined, size: 16, color: ConsoleColors.accent),
              const SizedBox(width: 8),
              const Expanded(
                child: Text(
                  '对话 / 文本交互',
                  style: TextStyle(
                    color: ConsoleColors.ink,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: statusColor.withValues(alpha: .12),
                  borderRadius: BorderRadius.circular(999),
                  border: Border.all(color: statusColor.withValues(alpha: .4)),
                ),
                child: Text(
                  statusLabel,
                  style: TextStyle(color: statusColor, fontSize: 11),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 16,
            runSpacing: 8,
            children: [
              _kv('模式', c.gestureMode ? '手势交互' : '文本指令'),
              _kv('后端', c.backend ? '已连接' : '未连接'),
              _kv('机器人', '${c.robotModelLabel} · ${c.isHardware ? '真机' : '模拟'}'),
              _kv('技能状态', c.skillStatus),
              _kv('当前技能', c.skillName),
              if (c.busy) _kv('进度', '${c.progress}%'),
              if (c.busy) _kv('耗时', '${c.modelDuration.inMilliseconds / 1000}s'),
              if (c.currentTask.isNotEmpty) _kv('当前任务', c.currentTask),
            ],
          ),
          if (c.busy) ...[
            const SizedBox(height: 10),
            ClipRRect(
              borderRadius: BorderRadius.circular(3),
              child: LinearProgressIndicator(
                value: (c.progress / 100).clamp(0, 1).toDouble(),
                minHeight: 4,
                backgroundColor: ConsoleColors.bg2,
                color: ConsoleColors.accent,
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _kv(String k, String v) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '$k ',
          style: const TextStyle(color: ConsoleColors.dim, fontSize: 11),
        ),
        Flexible(
          child: Text(
            v,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(color: ConsoleColors.muted, fontSize: 11),
          ),
        ),
      ],
    );
  }
}

class _ChatBody extends StatelessWidget {
  const _ChatBody({required this.controller, required this.scroll});

  final ConsoleController controller;
  final ScrollController scroll;

  @override
  Widget build(BuildContext context) {
    final turns = controller.dialog;
    if (turns.isEmpty) {
      return executionPanel(
        height: double.infinity,
        title: coloredTitle(
          Icons.chat_bubble_outline_rounded,
          '对话记录',
          Colors.transparent,
          ConsoleColors.cyan,
        ),
        trailing: tag('0'),
        child: emptyState(
          Icons.chat_outlined,
          '还没有对话',
          '在下方输入指令并发送；或回到任务页用文本模式发送',
        ),
      );
    }
    return executionPanel(
      height: double.infinity,
      title: coloredTitle(
        Icons.chat_bubble_outline_rounded,
        '对话记录',
        Colors.transparent,
        ConsoleColors.cyan,
      ),
      trailing: tag('${turns.length}'),
      child: ListView.builder(
        controller: scroll,
        padding: const EdgeInsets.only(bottom: 4),
        itemCount: turns.length,
        itemBuilder: (context, index) {
          final turn = turns[index];
          final isUser = turn.isUser;
          return Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: Column(
              crossAxisAlignment:
                  isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
              children: [
                Text(
                  '${isUser ? '你' : '机器人'}  ${turn.time}',
                  style: const TextStyle(
                    color: ConsoleColors.faint,
                    fontSize: 10,
                    fontFamily: 'monospace',
                  ),
                ),
                const SizedBox(height: 4),
                Align(
                  alignment: isUser
                      ? Alignment.centerRight
                      : Alignment.centerLeft,
                  child: Container(
                    constraints: const BoxConstraints(maxWidth: 640),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 9,
                    ),
                    decoration: BoxDecoration(
                      color: isUser ? ConsoleColors.field : ConsoleColors.bg2,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(
                        color: (isUser
                                ? ConsoleColors.accent
                                : ConsoleColors.lineSoft)
                            .withValues(alpha: .55),
                      ),
                    ),
                    child: Text(
                      turn.text,
                      style: const TextStyle(
                        color: ConsoleColors.ink,
                        fontSize: 12.5,
                        height: 1.45,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}

class _Composer extends StatelessWidget {
  const _Composer({
    required this.controller,
    required this.input,
    required this.onSend,
    required this.sessionActive,
    required this.onStart,
    required this.onStop,
  });

  final ConsoleController controller;
  final TextEditingController input;
  final VoidCallback onSend;
  final bool sessionActive;
  final VoidCallback onStart;
  final VoidCallback onStop;

  @override
  Widget build(BuildContext context) {
    final canSend =
        controller.backend && !controller.busy && input.text.trim().isNotEmpty;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        TextField(
          controller: input,
          minLines: 2,
          maxLines: 4,
          onSubmitted: (_) => onSend(),
          onChanged: (_) => controller.refresh(),
          style: const TextStyle(
            color: ConsoleColors.ink,
            fontSize: 13,
            height: 1.45,
          ),
          cursorColor: ConsoleColors.accent,
          decoration: InputDecoration(
            hintText: controller.busy
                ? '任务执行中…'
                : sessionActive
                ? '对话中…输入指令后点「发送文字指令」'
                : '先点「开始文本对话」，或直接输入指令发送',
            fillColor: ConsoleColors.field,
          ),
        ),
        const SizedBox(height: 10),
        SizedBox(
          width: double.infinity,
          height: 42,
          child: controller.busy
              ? OutlinedButton.icon(
                  onPressed: onStop,
                  icon: const Icon(Icons.stop_rounded, size: 18),
                  label: const Text('停止执行'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: ConsoleColors.red,
                    backgroundColor: ConsoleColors.red.withValues(alpha: .08),
                    side: BorderSide(
                      color: ConsoleColors.red.withValues(alpha: .55),
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(6),
                    ),
                  ),
                )
              : FilledButton.icon(
                  onPressed: canSend
                      ? onSend
                      : (sessionActive ? null : onStart),
                  iconAlignment: IconAlignment.end,
                  icon: const Icon(Icons.play_arrow_rounded, size: 18),
                  label: Text(
                    canSend
                        ? '发送文字指令'
                        : sessionActive
                        ? '对话中'
                        : '开始文本对话',
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  style: FilledButton.styleFrom(
                    backgroundColor: ConsoleColors.accent,
                    foregroundColor: ConsoleColors.bg0,
                    disabledBackgroundColor: ConsoleColors.bg2,
                    disabledForegroundColor: ConsoleColors.faint,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(6),
                    ),
                  ),
                ),
        ),
        const SizedBox(height: 8),
        Center(
          child: Text(
            !controller.backend
                ? '请先启动后端服务'
                : controller.busy
                ? '任务执行中，可在此停止'
                : sessionActive
                ? '文字交互已开启 · 回复见对话区，日志在下方'
                : '点「开始文本对话」；回复见对话区，日志在下方',
            textAlign: TextAlign.center,
            style: const TextStyle(color: ConsoleColors.dim, fontSize: 11),
          ),
        ),
      ],
    );
  }
}
