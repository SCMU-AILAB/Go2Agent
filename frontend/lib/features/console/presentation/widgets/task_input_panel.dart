import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';
import 'console_widgets.dart';

class TaskInputPanel extends StatelessWidget {
  const TaskInputPanel({super.key, required this.controller});

  final ConsoleController controller;

  @override
  Widget build(BuildContext context) {
    return panel(
      header: sectionTitle(Icons.terminal_outlined, 'TASK'),
      trailing: const Text(
        'Ctrl ↵',
        style: TextStyle(
          color: ConsoleColors.faint,
          fontSize: 11,
          fontFamily: 'monospace',
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(14, 14, 14, 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Wrap(
              spacing: 8,
              children: [
                ChoiceChip(
                  label: const Text('文本指令'),
                  selected: !controller.gestureMode,
                  onSelected: controller.busy
                      ? null
                      : (_) => controller.setTaskMode('text'),
                ),
                ChoiceChip(
                  label: const Text('持续视觉交互'),
                  selected: controller.gestureMode,
                  onSelected: controller.busy
                      ? null
                      : (_) => controller.setTaskMode('gesture'),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Align(
              alignment: Alignment.centerLeft,
              child: Text(
                controller.gestureMode
                    ? '视觉模式 · 持续观察和反馈，直到停止任务'
                    : controller.busy
                    ? '文本模式 · 任务处理中，状态见下方进度与 AGENT OUTPUT'
                    : '文本模式 · 点「发送指令」后，进度在本面板，回复在 AGENT OUTPUT',
                style: const TextStyle(color: ConsoleColors.dim, fontSize: 11),
              ),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: controller.taskController,
              minLines: 3,
              maxLines: 5,
              style: const TextStyle(
                color: ConsoleColors.ink,
                fontSize: 13,
                height: 1.45,
              ),
              cursorColor: ConsoleColors.accent,
              decoration: InputDecoration(
                hintText: controller.gestureMode
                    ? (controller.robotModel == 'GO2'
                          ? '描述目标或触发条件（例如：有人靠近就退后，看到比耶就比心）。模型会持续观察并选择已注册技能。'
                          : '持续观察画面，按任务提示与技能目录回应；不执行任意文本动作。')
                    : '告诉机器人要做什么…',
                fillColor: ConsoleColors.field,
              ),
            ),
            const SizedBox(height: 12),
            hudLabel('SUGGESTED TASKS'),
            const SizedBox(height: 7),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                if (controller.gestureMode)
                  _suggestion('社交互动', '用户给你打招呼的时候就给他打招呼，用户给你比耶或比心的时候你就比心。')
                else ...[
                  _suggestion('比心', '给我比个心。'),
                  _suggestion('站起来', '站起来。'),
                  _suggestion('坐下', '坐下。'),
                  _suggestion('趴下', '趴下休息。'),
                  _suggestion('伸懒腰', '伸个懒腰。'),
                  _suggestion('开心', '开心地动一下。'),
                  if (controller.robotModel == 'GO2') ...[
                    _suggestion('跳舞', '跳一段舞蹈一。'),
                    _suggestion('舞蹈二', '跳第二段舞。'),
                    _suggestion('作揖', '作个揖。'),
                  ],
                  _suggestion('向前移动', '站稳后向前移动 0.2 米，然后停止。'),
                  _suggestion('后退', '站稳后后退 0.2 米，然后停止。'),
                  _suggestion('左转', '站稳后向左转 15 度，然后停止。'),
                  _suggestion('右转', '站稳后向右转 15 度，然后停止。'),
                  _suggestion('打招呼', '打个招呼。'),
                ],
              ],
            ),
            const SizedBox(height: 14),
            SizedBox(
              width: double.infinity,
              height: 42,
              child: controller.busy
                  ? OutlinedButton.icon(
                      onPressed: controller.cancelTask,
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
                      onPressed:
                          controller.backend &&
                              (!controller.gestureMode ||
                                  controller.cameraSource == 'local') &&
                              controller.taskController.text.trim().isNotEmpty
                          ? controller.submitTask
                          : null,
                      iconAlignment: IconAlignment.end,
                      icon: const Icon(Icons.play_arrow_rounded, size: 18),
                      label: Text(
                        controller.gestureMode
                            ? '开始视觉交互'
                            : '开始文本对话',
                      ),
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
                          fontFamily: 'PingFang SC',
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
                    ? '任务执行中，可按 Esc 停止'
                    : controller.gestureMode
                    ? (controller.cameraSource != 'local'
                          ? '请先选择本地相机；模拟画面不能用于视觉决策'
                          : controller.robotModel == 'GO2'
                          ? '按任务提示词自主决策；仅执行技能目录中的动作'
                          : '仅握手 / 挥手 / 击掌；持续运行至停止')
                    : '文本模式看 AGENT OUTPUT 与下方进度条；视觉模式看持续决策流',
                textAlign: TextAlign.center,
                style: const TextStyle(color: ConsoleColors.dim, fontSize: 11),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _suggestion(String label, String prompt) {
    return OutlinedButton(
      onPressed: () {
        controller.taskController.text = prompt;
        controller.taskController.selection = TextSelection.collapsed(
          offset: prompt.length,
        );
      },
      style: OutlinedButton.styleFrom(
        foregroundColor: ConsoleColors.muted,
        backgroundColor: ConsoleColors.field,
        side: const BorderSide(color: ConsoleColors.line),
        padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
        minimumSize: const Size(0, 28),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
        textStyle: const TextStyle(fontSize: 11, fontFamily: 'PingFang SC'),
      ),
      child: Text(label),
    );
  }
}
