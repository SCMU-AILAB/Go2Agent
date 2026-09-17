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
                  label: const Text('持续手势交互'),
                  selected: controller.gestureMode,
                  onSelected: controller.busy
                      ? null
                      : (_) => controller.setTaskMode('gesture'),
                ),
              ],
            ),
            const SizedBox(height: 10),
            if (controller.gestureMode && controller.robotModel == 'GO2') ...[
              Wrap(
                spacing: 6,
                children: [
                  ChoiceChip(
                    label: const Text('挥手后打招呼'),
                    selected: controller.waveResponse == 'wave',
                    onSelected: controller.busy
                        ? null
                        : (_) => controller.setWaveResponse('wave'),
                  ),
                  ChoiceChip(
                    label: const Text('挥手后比心'),
                    selected: controller.waveResponse == 'heart',
                    onSelected: controller.busy
                        ? null
                        : (_) => controller.setWaveResponse('heart'),
                  ),
                ],
              ),
              const SizedBox(height: 10),
            ],
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
                          ? '持续观察挥手，按上方选定动作回应。握手/击掌不执行。'
                          : '持续观察握手、挥手、击掌；不执行文字动作命令。')
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
                  _suggestion('回应挥手', '持续观察画面，只在确认有人向我挥手时按选定动作回应。')
                else ...[
                  _suggestion('比心', '给我比个心。'),
                  _suggestion('站起来', '站起来。'),
                  _suggestion('坐下', '坐下。'),
                  _suggestion('向前移动', '站稳后向前移动 0.2 米，然后停止。'),
                  if (controller.robotModel == 'GO2')
                    _suggestion('跳舞', '跳一段舞蹈一。'),
                ],
                if (!controller.gestureMode) _suggestion('打招呼', '打个招呼。'),
              ],
            ),
            const SizedBox(height: 14),
            SizedBox(
              width: double.infinity,
              height: 42,
              child: FilledButton.icon(
                onPressed:
                    controller.backend &&
                        !controller.busy &&
                        (!controller.gestureMode ||
                            controller.cameraSource == 'local') &&
                        controller.taskController.text.trim().isNotEmpty
                    ? controller.submitTask
                    : null,
                iconAlignment: IconAlignment.end,
                icon: const Icon(Icons.play_arrow_rounded, size: 18),
                label: Text(controller.gestureMode ? '开始持续视觉交互' : '发送指令'),
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
                          ? '请先选择本地相机；模拟画面不能识别手势'
                          : controller.robotModel == 'GO2'
                          ? '确认挥手后按所选动作回应；直接执行动作请选文本指令'
                          : '仅握手 / 挥手 / 击掌；持续运行至停止')
                    : '可保留真实相机预览；文本模式不看图、不提供视觉避障',
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
