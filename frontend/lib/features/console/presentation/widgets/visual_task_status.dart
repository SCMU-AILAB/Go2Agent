import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';

class VisualTaskStatus extends StatelessWidget {
  const VisualTaskStatus({super.key, required this.controller});

  final ConsoleController controller;

  @override
  Widget build(BuildContext context) {
    final task = controller.visionTask;
    final spec = task['spec'] is Map ? task['spec'] as Map : const {};
    final arguments = spec['skill_arguments'] is Map
        ? spec['skill_arguments'] as Map
        : const {};
    final follow = arguments['follow_person'] is Map
        ? arguments['follow_person'] as Map
        : const {};
    final observation = controller.cameraObservation;
    final tracking = task['tracking'] is Map
        ? task['tracking'] as Map
        : const {};
    final state = switch (task['state']) {
      'planning' => '理解任务',
      'observing' => '观察与决策',
      'executing' => '技能执行中',
      'blocked' => '任务受阻',
      'waiting_model' => '等待模型恢复',
      'stopped' => '已停止',
      'failed' => '任务失败',
      _ => '等待视觉任务',
    };
    final distance = observation['nearest_person_distance_m'];
    final metrics = task['metrics'] is Map ? task['metrics'] as Map : const {};
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: ConsoleColors.field,
        border: Border.all(color: ConsoleColors.line),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('视觉任务状态：$state'),
          if (spec['goal'] != null) Text('目标：${spec['goal']}'),
          if (spec['target'] != null) Text('对象：${spec['target']}'),
          if (follow.isNotEmpty)
            Text('保持距离：${follow['target_distance_m'] ?? 1.5} 米'),
          Text(
            '人员数：${observation['person_count'] ?? '未知'} · '
            '距离：${distance is num ? '${distance.toStringAsFixed(2)} 米' : '无有效深度'}',
          ),
          if (tracking['reason'] != null) Text('跟随感知：${tracking['reason']}'),
          if (tracking['ageS'] is num)
            Text('人员数据年龄：${(tracking['ageS'] as num).toStringAsFixed(2)} 秒'),
          if (task['activeSkill'] != null) Text('当前技能：${task['activeSkill']}'),
          if (task['reason'] != null) Text('原因：${task['reason']}'),
          if (controller.cameraError != null)
            Text('相机：${controller.cameraError}'),
          if (task['frameAgeS'] is num)
            Text('决策画面年龄：${(task['frameAgeS'] as num).toStringAsFixed(2)} 秒'),
          if (metrics['round_trip_s'] is num)
            Text(
              '模型耗时：${(metrics['round_trip_s'] as num).toStringAsFixed(2)} 秒',
            ),
        ],
      ),
    );
  }
}
