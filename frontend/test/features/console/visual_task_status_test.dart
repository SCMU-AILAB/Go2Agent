import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:g1_frontend/features/console/controllers/console_controller.dart';
import 'package:g1_frontend/features/console/presentation/widgets/task_input_panel.dart';

import '../../support/fake_console_api.dart';

void main() {
  testWidgets(
    'visual task shows target distance and model wait reason on phone',
    (tester) async {
      final controller = ConsoleController(
        onMessage: (_) {},
        api: FakeConsoleApi(),
      );
      controller.setTaskMode('vision');
      controller.backend = true;
      controller.cameraSource = 'local';
      controller.cameraObservation = {
        'person_count': 1,
        'nearest_person_distance_m': 2.1,
      };
      controller.visionTask = {
        'state': 'waiting_model',
        'reason': 'shared model HTTP 429',
        'spec': {
          'goal': '跟随人员',
          'target': '单个可见人员',
          'skill_arguments': {
            'follow_person': {'target_distance_m': 2.0},
          },
        },
      };
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SingleChildScrollView(
              child: SizedBox(
                width: 320,
                child: TaskInputPanel(controller: controller),
              ),
            ),
          ),
        ),
      );
      expect(find.text('持续视觉任务'), findsOneWidget);
      expect(find.text('视觉任务状态：等待模型恢复'), findsOneWidget);
      expect(find.text('保持距离：2.0 米'), findsOneWidget);
      expect(find.textContaining('无有效深度'), findsNothing);
      expect(find.text('原因：shared model HTTP 429'), findsOneWidget);
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox());
      controller.dispose();
    },
  );

  testWidgets('running task offers cancellation and replacement', (
    tester,
  ) async {
    final api = FakeConsoleApi();
    final controller = ConsoleController(onMessage: (_) {}, api: api);
    controller.setTaskMode('vision');
    controller.backend = true;
    controller.busy = true;
    controller.cameraSource = 'local';
    controller.taskController.text = '看到人就坐下';
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: TaskInputPanel(controller: controller),
          ),
        ),
      ),
    );
    expect(find.text('停止执行'), findsOneWidget);
    await tester.ensureVisible(find.text('停止旧任务并应用新目标'));
    await tester.tap(find.text('停止旧任务并应用新目标'));
    await tester.pump();
    expect(api.lastTaskMode, 'vision');
    expect(api.lastReplaceExisting, isTrue);
    await tester.pumpWidget(const SizedBox());
    controller.dispose();
  });
}
