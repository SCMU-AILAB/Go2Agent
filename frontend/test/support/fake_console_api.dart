import 'dart:async';
import 'dart:typed_data';

import 'package:g1_frontend/features/console/models/console_snapshot.dart';
import 'package:g1_frontend/features/console/services/console_api.dart';

Map<String, dynamic> consolePayload({
  bool backend = false,
  bool starting = false,
  bool busy = false,
  bool promptSaved = true,
  String systemPrompt = '测试系统提示词',
  String sessionId = 'TEST01',
  String? taskId,
  String cameraSource = 'demo',
  String modelStatus = '待命',
  String skillStatus = 'IDLE',
  String skillName = '等待调度',
  int progress = 0,
  String progressText = '等待执行',
  int activeStep = -1,
  String currentTask = '',
  String modelOutput = '',
  double modelDuration = 0,
  int latency = 4,
  int taskCount = 0,
  String robotMode = 'simulation',
  bool robotConnected = true,
  String cameraLabel = '模拟视频源',
  String cameraStatus = 'ready',
  bool cameraFrameAvailable = true,
  String cameraFrameUrl = '/api/v1/camera/frame.jpg',
  int cameraFrameVersion = 0,
  int cameraWidth = 640,
  int cameraHeight = 480,
  int cameraFps = 30,
  String? cameraError,
  bool voiceEnabled = false,
  bool voiceListening = false,
  String voiceStatus = 'stopped',
  String voiceTranscript = '',
  String voiceReply = '',
  String? voiceError,
  String voiceInputDevice = 'pulse',
  String voiceOutputDevice = 'pulse',
  String? voiceTtsEngine = 'espeak-ng',
  List<Map<String, dynamic>> tools = const [],
  List<Map<String, dynamic>> logs = const [],
}) => {
  'backend': backend,
  'starting': starting,
  'busy': busy,
  'promptSaved': promptSaved,
  'systemPrompt': systemPrompt,
  'sessionId': sessionId,
  'taskId': taskId,
  'cameraSource': cameraSource,
  'modelStatus': modelStatus,
  'skillStatus': skillStatus,
  'skillName': skillName,
  'progress': progress,
  'progressText': progressText,
  'activeStep': activeStep,
  'currentTask': currentTask,
  'modelOutput': modelOutput,
  'modelDuration': modelDuration,
  'latency': latency,
  'taskCount': taskCount,
  'robot': {
    'mode': robotMode,
    'connected': robotConnected,
    'details': <String, Object?>{},
  },
  'camera': {
    'source': cameraSource,
    'label': cameraLabel,
    'status': cameraStatus,
    'frameAvailable': cameraFrameAvailable,
    'frameUrl': cameraFrameUrl,
    'frameVersion': cameraFrameVersion,
    'width': cameraWidth,
    'height': cameraHeight,
    'fps': cameraFps,
    'error': cameraError,
  },
  'voice': {
    'enabled': voiceEnabled,
    'listening': voiceListening,
    'status': voiceStatus,
    'transcript': voiceTranscript,
    'reply': voiceReply,
    'error': voiceError,
    'inputDevice': voiceInputDevice,
    'outputDevice': voiceOutputDevice,
    'ttsEngine': voiceTtsEngine,
  },
  'tools': tools,
  'logs': logs,
};

class FakeConsoleApi implements ConsoleApi {
  FakeConsoleApi({Map<String, dynamic>? initialPayload})
    : _payload = initialPayload ?? consolePayload();

  @override
  final Uri baseUri = Uri.parse('http://g1.test:8000');

  final StreamController<Map<String, dynamic>> _events =
      StreamController<Map<String, dynamic>>.broadcast();
  Map<String, dynamic> _payload;

  int getConsoleCalls = 0;
  int startSessionCalls = 0;
  int stopSessionCalls = 0;
  int updatePromptCalls = 0;
  int submitTaskCalls = 0;
  String? lastTaskMode;
  int cancelTaskCalls = 0;
  int emergencyStopCalls = 0;
  int setCameraSourceCalls = 0;
  int clearLogsCalls = 0;
  int fetchCameraFrameCalls = 0;
  int startVoiceCalls = 0;
  int stopVoiceCalls = 0;
  int speakCalls = 0;
  String? lastSpeech;
  bool closeCalled = false;
  String? lastPrompt;
  String? lastInstruction;
  String? lastTaskCameraSource;
  String? lastCancelReason;

  ConsoleSnapshot get snapshot => ConsoleSnapshot.fromJson(_payload);

  void replacePayload(Map<String, dynamic> payload) {
    _payload = payload;
  }

  void emit(String type, Map<String, dynamic> data) {
    _events.add({'type': type, 'data': data});
  }

  void emitState(Map<String, dynamic> payload) {
    replacePayload(payload);
    emit('state', payload);
  }

  @override
  Future<ConsoleSnapshot> getConsole() async {
    getConsoleCalls += 1;
    return snapshot;
  }

  @override
  Future<ConsoleSnapshot> startSession() async {
    startSessionCalls += 1;
    _payload = {
      ..._payload,
      'backend': true,
      'starting': false,
      'sessionId': 'LIVE01',
      'modelStatus': '待命',
      'skillStatus': 'IDLE',
    };
    return snapshot;
  }

  @override
  Future<ConsoleSnapshot> stopSession() async {
    stopSessionCalls += 1;
    _payload = {
      ..._payload,
      'backend': false,
      'starting': false,
      'busy': false,
      'modelStatus': '已停止',
      'skillStatus': 'STOPPED',
    };
    return snapshot;
  }

  @override
  Future<ConsoleSnapshot> updateSystemPrompt(String prompt) async {
    updatePromptCalls += 1;
    lastPrompt = prompt;
    _payload = {..._payload, 'systemPrompt': prompt, 'promptSaved': true};
    return snapshot;
  }

  @override
  Future<ConsoleSnapshot> submitTask(
    String instruction, {
    String? cameraSource,
    String? taskMode,
  }) async {
    submitTaskCalls += 1;
    lastInstruction = instruction;
    lastTaskCameraSource = cameraSource;
    lastTaskMode = taskMode;
    _payload = {
      ..._payload,
      'busy': true,
      'currentTask': instruction,
      'modelStatus': '生成中',
      'skillStatus': 'RUNNING',
      'progress': 10,
      'progressText': '正在感知环境',
      'activeStep': 0,
    };
    return snapshot;
  }

  @override
  Future<ConsoleSnapshot> cancelTask(String reason) async {
    cancelTaskCalls += 1;
    lastCancelReason = reason;
    _payload = {
      ..._payload,
      'busy': false,
      'modelStatus': '已停止',
      'skillStatus': 'STOPPED',
      'progressText': '任务已停止',
    };
    return snapshot;
  }

  @override
  Future<ConsoleSnapshot> emergencyStop() async {
    emergencyStopCalls += 1;
    _payload = {
      ..._payload,
      'busy': false,
      'modelStatus': '急停',
      'skillStatus': 'STOPPED',
      'progressText': '已急停',
    };
    return snapshot;
  }

  @override
  Future<ConsoleSnapshot> setCameraSource(String source) async {
    setCameraSourceCalls += 1;
    final local = source == 'local';
    final camera = Map<String, dynamic>.from(_payload['camera'] as Map);
    camera
      ..['source'] = source
      ..['label'] = local ? 'USB RealSense D435i' : '模拟视频源'
      ..['status'] = local ? 'starting' : 'ready'
      ..['frameAvailable'] = !local;
    _payload = {..._payload, 'cameraSource': source, 'camera': camera};
    return snapshot;
  }

  @override
  Future<ConsoleSnapshot> clearLogs() async {
    clearLogsCalls += 1;
    _payload = {..._payload, 'logs': <Map<String, dynamic>>[]};
    return snapshot;
  }

  @override
  Future<ConsoleSnapshot> startVoice() async {
    startVoiceCalls += 1;
    _payload = {
      ..._payload,
      'voice': {
        ...Map<String, dynamic>.from(_payload['voice'] as Map),
        'enabled': true,
        'listening': true,
        'status': 'listening',
      },
    };
    return snapshot;
  }

  @override
  Future<ConsoleSnapshot> stopVoice() async {
    stopVoiceCalls += 1;
    _payload = {
      ..._payload,
      'voice': {
        ...Map<String, dynamic>.from(_payload['voice'] as Map),
        'enabled': false,
        'listening': false,
        'status': 'stopped',
      },
    };
    return snapshot;
  }

  @override
  Future<ConsoleSnapshot> speak(String text) async {
    speakCalls += 1;
    lastSpeech = text;
    _payload = {
      ..._payload,
      'voice': {
        ...Map<String, dynamic>.from(_payload['voice'] as Map),
        'reply': text,
      },
    };
    return snapshot;
  }

  @override
  Future<Uint8List> fetchCameraFrame(String path) async {
    fetchCameraFrameCalls += 1;
    return Uint8List(0);
  }

  @override
  Stream<Map<String, dynamic>> events() => _events.stream;

  @override
  Uri cameraFrameUri(String path, int version) =>
      baseUri.resolve(path).replace(queryParameters: {'v': '$version'});

  @override
  Future<void> close() async {
    closeCalled = true;
  }

  Future<void> dispose() => _events.close();
}
