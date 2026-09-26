import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go2_frontend/app.dart';

import 'support/fake_console_api.dart';

void expectNoLayoutException(WidgetTester tester) {
  final exception = tester.takeException();
  if (exception == null) return;
  for (final element in find.bySubtype<Flex>().evaluate()) {
    final render = element.renderObject;
    if (render is! RenderFlex || !render.hasSize) continue;
    var maxX = 0.0;
    var maxY = 0.0;
    RenderBox? child = render.firstChild;
    while (child != null) {
      final data = child.parentData! as FlexParentData;
      maxX = maxX > data.offset.dx + child.size.width
          ? maxX
          : data.offset.dx + child.size.width;
      maxY = maxY > data.offset.dy + child.size.height
          ? maxY
          : data.offset.dy + child.size.height;
      child = render.childAfter(child);
    }
    if (maxX > render.size.width + .1 || maxY > render.size.height + .1) {
      debugPrint(
        'OVERFLOW CANDIDATE ${element.widget} size=${render.size} children=($maxX,$maxY)',
      );
    }
  }
  fail(exception.toString());
}

void main() {
  testWidgets('renders FastAPI state on a desktop viewport', (tester) async {
    tester.view.devicePixelRatio = 1;
    tester.view.physicalSize = const Size(1440, 1000);
    final api = FakeConsoleApi(
      initialPayload: consolePayload(
        backend: true,
        sessionId: 'LIVE01',
        robotMode: 'hardware',
        robotConnected: true,
      ),
    );
    addTearDown(() async {
      tester.view.resetDevicePixelRatio();
      tester.view.resetPhysicalSize();
      await api.dispose();
    });

    await tester.pumpWidget(Go2ConsoleApp(api: api));
    await tester.pump();

    expect(find.text('机器人控制台'), findsOneWidget);
    expect(find.text('停止后端'), findsOneWidget);
    expect(find.text('HARDWARE · 真机模式'), findsOneWidget);
    expect(find.text('SESSION  LIVE01'), findsOneWidget);
    expect(find.text('AGENT OUTPUT'), findsOneWidget);
    expect(find.text('TOOL CALLS'), findsOneWidget);
    expect(find.text('SKILL PIPELINE'), findsOneWidget);
    expect(find.text('ROBOT STATUS'), findsOneWidget);
    expectNoLayoutException(tester);
  });

  testWidgets('expands the robot status panel', (tester) async {
    tester.view.devicePixelRatio = 1;
    tester.view.physicalSize = const Size(1440, 1000);
    final api = FakeConsoleApi(
      initialPayload: consolePayload(backend: true, robotConnected: true),
    );
    addTearDown(() async {
      tester.view.resetDevicePixelRatio();
      tester.view.resetPhysicalSize();
      await api.dispose();
    });

    await tester.pumpWidget(Go2ConsoleApp(api: api));
    await tester.pump();

    AnimatedCrossFade fade() =>
        tester.widget<AnimatedCrossFade>(find.byType(AnimatedCrossFade));

    // Collapsed by default: the operator's main job is issuing tasks.
    expect(fade().crossFadeState, CrossFadeState.showFirst);
    await tester.ensureVisible(find.text('ROBOT STATUS'));
    await tester.pump();
    await tester.tap(find.text('ROBOT STATUS'));
    await tester.pump();
    expect(fade().crossFadeState, CrossFadeState.showSecond);
    await tester.pump(const Duration(milliseconds: 300));
    expect(find.text('ONBOARD TELEMETRY'), findsOneWidget);
    expectNoLayoutException(tester);
  });

  testWidgets('adapts to a phone viewport without layout errors', (
    tester,
  ) async {
    tester.view.devicePixelRatio = 1;
    tester.view.physicalSize = const Size(390, 844);
    final api = FakeConsoleApi();
    addTearDown(() async {
      tester.view.resetDevicePixelRatio();
      tester.view.resetPhysicalSize();
      await api.dispose();
    });

    await tester.pumpWidget(Go2ConsoleApp(api: api));
    await tester.pump();

    expect(find.text('机器人控制台'), findsOneWidget);
    expect(find.text('SIM'), findsOneWidget);
    expectNoLayoutException(tester);
  });

  testWidgets('submits a task and renders websocket completion', (
    tester,
  ) async {
    tester.view.devicePixelRatio = 1;
    tester.view.physicalSize = const Size(1440, 1000);
    final api = FakeConsoleApi(initialPayload: consolePayload(backend: true));
    addTearDown(() async {
      tester.view.resetDevicePixelRatio();
      tester.view.resetPhysicalSize();
      await api.dispose();
    });

    await tester.pumpWidget(Go2ConsoleApp(api: api));
    await tester.pump();
    await tester.ensureVisible(find.text('打招呼'));
    await tester.pump();
    await tester.tap(find.text('打招呼'));
    await tester.pump();
    await tester.ensureVisible(find.text('开始文本对话'));
    await tester.pump();
    await tester.tap(find.text('开始文本对话'));
    await tester.pump();

    expect(api.submitTaskCalls, 1);
    api.emitState(
      consolePayload(
        backend: true,
        modelStatus: '已完成',
        skillStatus: 'DONE',
        skillName: 'wave',
        progress: 100,
        progressText: '执行完成',
        activeStep: 3,
        currentTask: '向我挥手打个招呼。',
        modelOutput: '好的，你好！',
        taskCount: 1,
        tools: const [
          {
            'name': 'wave',
            'payload': '{"success": true}',
            'arguments': {'arm': 'right'},
            'result': {'success': true},
          },
        ],
      ),
    );
    await tester.pump();

    // modelStatus shows both in the execution stream and in the SYSTEM panel.
    expect(find.text('已完成'), findsWidgets);
    expect(find.text('wave'), findsWidgets);
    expect(find.text('好的，你好！'), findsOneWidget);
    expectNoLayoutException(tester);
  });
}
