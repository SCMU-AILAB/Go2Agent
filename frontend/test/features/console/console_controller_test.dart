import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:go2_frontend/features/console/controllers/console_controller.dart';
import 'package:go2_frontend/features/console/services/console_api.dart';

import '../../support/fake_console_api.dart';

void main() {
  late FakeConsoleApi api;
  late ConsoleController controller;
  late List<String> messages;

  setUp(() {
    messages = [];
    api = FakeConsoleApi(
      initialPayload: consolePayload(
        systemPrompt: '来自 FastAPI 的提示词',
        sessionId: 'API001',
        robotMode: 'hardware',
        robotConnected: true,
        cameraSource: 'local',
        cameraLabel: 'USB RealSense D435i',
        cameraStatus: 'ready',
        cameraFrameAvailable: true,
        cameraFrameVersion: 3,
        logs: const [
          {
            'id': 'log-1',
            'time': '12:00:00',
            'level': 'INFO',
            'source': 'backend',
            'message': 'API 已就绪',
          },
        ],
      ),
    );
    controller = ConsoleController(onMessage: messages.add, api: api)
      ..initialize();
  });

  tearDown(() async {
    if (controller.isActive) controller.dispose();
    await api.dispose();
  });

  testWidgets('task mode is explicit and demo blocks gesture work', (
    tester,
  ) async {
    controller.backend = true;
    controller.taskController.text = '回应挥手';
    controller.setTaskMode('gesture');
    controller.cameraSource = 'demo';
    await controller.submitTask();
    expect(api.submitTaskCalls, 0);
    expect(messages.last, contains('本地相机'));
    controller.cameraSource = 'local';
    controller.robotModel = 'GO2';
    await controller.submitTask();
    expect(api.lastTaskMode, 'gesture');
    controller.setTaskMode('text');
    expect(controller.taskMode, 'gesture');
  });

  testWidgets('initializes from the FastAPI console snapshot', (tester) async {
    await tester.pump();
    await controller.refreshFromBackend();

    expect(api.getConsoleCalls, greaterThanOrEqualTo(1));
    expect(controller.connectedToApi, isTrue);
    expect(controller.sessionId, 'API001');
    expect(controller.systemPromptController.text, '来自 FastAPI 的提示词');
    expect(controller.isHardware, isTrue);
    expect(controller.cameraSource, 'local');
    expect(controller.cameraFrameVersion, 3);
    expect(controller.logs.single.message, 'API 已就绪');
  });

  testWidgets('starts and stops the backend through REST', (tester) async {
    await tester.pump();

    await controller.toggleBackend();
    expect(api.startSessionCalls, 1);
    expect(controller.backend, isTrue);
    expect(messages.last, '后端已启动');

    await controller.toggleBackend();
    expect(api.stopSessionCalls, 1);
    expect(controller.backend, isFalse);
    expect(messages.last, '后端已停止');
  });

  testWidgets('failed emergency stop never reports success', (tester) async {
    await tester.pump();
    api.emergencyStopError = const ConsoleApiException(
      'Go2 stop command failed', statusCode: 502,
    );

    await controller.emergencyStop();

    expect(api.emergencyStopCalls, 1);
    expect(api.cancelTaskCalls, 0);
    expect(messages.last, contains('急停失败'));
  });

  testWidgets('saves prompt and submits and cancels a task through REST', (
    tester,
  ) async {
    await tester.pump();
    await controller.toggleBackend();

    controller.systemPromptController.text = '只执行安全技能';
    expect(controller.promptSaved, isFalse);
    await controller.savePrompt();
    expect(api.updatePromptCalls, 1);
    expect(api.lastPrompt, '只执行安全技能');
    expect(controller.promptSaved, isTrue);

    controller.taskController.text = '向我挥手';
    await controller.submitTask();
    expect(api.submitTaskCalls, 1);
    expect(api.lastInstruction, '向我挥手');
    expect(api.lastTaskCameraSource, 'local');
    expect(api.lastTaskMode, 'text');
    expect(controller.busy, isTrue);

    await controller.cancelTask('测试中断');
    expect(api.cancelTaskCalls, 1);
    expect(api.lastCancelReason, '测试中断');
    expect(controller.busy, isFalse);
    expect(controller.skillStatus, 'STOPPED');
  });

  testWidgets('applies websocket state, log, heartbeat, and camera events', (
    tester,
  ) async {
    await tester.pump();

    api.emitState(
      consolePayload(
        backend: true,
        busy: true,
        currentTask: '观察前方',
        modelStatus: '生成中',
        skillStatus: 'RUNNING',
        progress: 70,
        progressText: '正在执行技能',
        activeStep: 2,
      ),
    );
    api.emit('log', {
      'id': 'log-live',
      'time': '12:00:01',
      'level': 'WARN',
      'source': 'camera',
      'message': '深度帧延迟',
    });
    api.emit('heartbeat', {'latencyMs': 17, 'connected': false});
    api.emit('camera', {'frameVersion': 9});
    await tester.pump();

    expect(controller.backend, isTrue);
    expect(controller.busy, isTrue);
    expect(controller.currentTask, '观察前方');
    expect(controller.progress, 70);
    expect(controller.logs.last.id, 'log-live');
    expect(controller.latency, 17);
    expect(controller.robotConnected, isFalse);
    expect(controller.cameraFrameVersion, 9);
    expect(controller.cameraFrameAvailable, isTrue);
  });

  testWidgets('switches camera and clears logs through REST', (tester) async {
    await tester.pump();

    await controller.selectCameraSource('demo');
    expect(api.setCameraSourceCalls, 1);
    expect(controller.cameraSource, 'demo');

    controller.addLog('INFO', 'test', '本地日志');
    await controller.clearLogs();
    expect(api.clearLogsCalls, 1);
    expect(controller.logs, isEmpty);
  });

  testWidgets('starts and stops local voice through REST', (tester) async {
    await tester.pump();
    await controller.refreshFromBackend();
    await controller.toggleBackend();

    await controller.toggleVoice();
    expect(api.startVoiceCalls, 1);
    expect(controller.voiceEnabled, isTrue);
    expect(controller.voiceListening, isTrue);

    await controller.toggleVoice();
    expect(api.stopVoiceCalls, 1);
    expect(controller.voiceEnabled, isFalse);
  });

  testWidgets('keeps the running log view pinned to the newest entry', (
    tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: ListenableBuilder(
          listenable: controller,
          builder: (context, child) => SizedBox(
            height: 80,
            child: ListView.builder(
              controller: controller.logScrollController,
              itemCount: controller.visibleLogs.length,
              itemBuilder: (context, index) => SizedBox(
                height: 24,
                child: Text(controller.visibleLogs[index].message),
              ),
            ),
          ),
        ),
      ),
    );
    await tester.pump();

    for (var index = 0; index < 30; index += 1) {
      api.emit('log', {
        'id': 'scroll-$index',
        'time': '12:00:01',
        'level': 'INFO',
        'source': 'test',
        'message': '日志 $index',
      });
    }
    await tester.pump();
    await tester.pump();

    final position = controller.logScrollController.position;
    expect(position.pixels, position.maxScrollExtent);
    expect(controller.logs.last.message, '日志 29');
  });

  testWidgets('keeps visual loop busy and displays its decision and outcome', (
    tester,
  ) async {
    await tester.pump();
    api.emitState(
      consolePayload(
        backend: true,
        busy: true,
        cameraSource: 'local',
        modelStatus: '持续视觉交互',
        modelOutput: '{"decision":{"skill":"wave","speech":"你好"}}',
        skillStatus: 'DONE',
        tools: const [
          {
            'name': 'vision.decide',
            'payload': '{"frame_count":3}',
            'arguments': {},
            'result': {'frame_count': 3},
          },
          {
            'name': 'vision.outcome',
            'payload': '{"speech_spoken":true}',
            'arguments': {},
            'result': {'speech_spoken': true},
          },
        ],
      ),
    );
    await tester.pump();
    expect(controller.busy, isTrue);
    expect(controller.modelStatus, '持续视觉交互');
    expect(controller.modelOutput, contains('你好'));
    expect(
      controller.tools.map((tool) => tool.name),
      containsAll(['vision.decide', 'vision.outcome']),
    );
    await controller.cancelTask('停止视觉交互');
    expect(api.cancelTaskCalls, 1);
    expect(controller.busy, isFalse);
  });

  testWidgets('dispose closes the client and ignores later events', (
    tester,
  ) async {
    await tester.pump();
    final oldLatency = controller.latency;
    controller.dispose();
    await tester.pump();

    api.emit('heartbeat', {'latencyMs': 999, 'connected': false});
    await tester.pump();

    expect(api.closeCalled, isTrue);
    expect(controller.latency, oldLatency);
  });
}
