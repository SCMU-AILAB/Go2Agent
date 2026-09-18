import 'console_log.dart';
import 'tool_call.dart';

class ConsoleSnapshot {
  const ConsoleSnapshot({
    required this.backend,
    required this.starting,
    required this.busy,
    required this.promptSaved,
    required this.systemPrompt,
    required this.sessionId,
    required this.cameraSource,
    required this.modelStatus,
    required this.skillStatus,
    required this.skillName,
    required this.progress,
    required this.progressText,
    required this.activeStep,
    required this.currentTask,
    required this.modelOutput,
    required this.modelDurationSeconds,
    required this.latency,
    required this.taskCount,
    required this.robotMode,
    required this.robotConnected,
    required this.robotModel,
    required this.telemetryAvailable,
    required this.cameraLabel,
    required this.cameraStatus,
    required this.cameraFrameAvailable,
    required this.cameraFramePath,
    required this.cameraFrameVersion,
    required this.cameraWidth,
    required this.cameraHeight,
    required this.cameraFps,
    required this.cameraError,
    required this.voiceEnabled,
    required this.voiceListening,
    required this.voiceStatus,
    required this.voiceTranscript,
    required this.voiceReply,
    required this.voiceError,
    required this.voiceInputDevice,
    required this.voiceOutputDevice,
    required this.voiceTtsEngine,
    required this.tools,
    required this.logs,
  });

  factory ConsoleSnapshot.fromJson(Map<String, dynamic> json) {
    final robot = _asMap(json['robot']);
    final camera = _asMap(json['camera']);
    final voice = _asMap(json['voice']);
    final robotDetails = _asMap(robot['details']);
    final rawModel = (robotDetails['robot_model'] as String?)?.toLowerCase();
    final robotModel = switch (rawModel) {
      'g1' => 'G1',
      'go2' => 'GO2',
      _ => null,
    };
    return ConsoleSnapshot(
      backend: json['backend'] as bool? ?? false,
      starting: json['starting'] as bool? ?? false,
      busy: json['busy'] as bool? ?? false,
      promptSaved: json['promptSaved'] as bool? ?? true,
      systemPrompt: json['systemPrompt'] as String? ?? '',
      sessionId: json['sessionId'] as String? ?? '—',
      cameraSource: json['cameraSource'] as String? ?? 'demo',
      modelStatus: json['modelStatus'] as String? ?? '待命',
      skillStatus: json['skillStatus'] as String? ?? 'IDLE',
      skillName: json['skillName'] as String? ?? '等待调度',
      progress: (json['progress'] as num?)?.toInt() ?? 0,
      progressText: json['progressText'] as String? ?? '等待执行',
      activeStep: (json['activeStep'] as num?)?.toInt() ?? -1,
      currentTask: json['currentTask'] as String? ?? '',
      modelOutput: json['modelOutput'] as String? ?? '',
      modelDurationSeconds: (json['modelDuration'] as num?)?.toDouble() ?? 0,
      latency: (json['latency'] as num?)?.toInt() ?? 0,
      taskCount: (json['taskCount'] as num?)?.toInt() ?? 0,
      robotMode: robot['mode'] as String? ?? 'simulation',
      robotConnected: robot['connected'] as bool? ?? false,
      robotModel: robotModel,
      telemetryAvailable: robotDetails['telemetry_available'] as bool?,
      cameraLabel: camera['label'] as String? ?? '模拟视频源',
      cameraStatus: camera['status'] as String? ?? 'idle',
      cameraFrameAvailable: camera['frameAvailable'] as bool? ?? false,
      cameraFramePath:
          camera['frameUrl'] as String? ?? '/api/v1/camera/frame.jpg',
      cameraFrameVersion: (camera['frameVersion'] as num?)?.toInt() ?? 0,
      cameraWidth: (camera['width'] as num?)?.toInt() ?? 640,
      cameraHeight: (camera['height'] as num?)?.toInt() ?? 480,
      cameraFps: (camera['fps'] as num?)?.toInt() ?? 30,
      cameraError: camera['error'] as String?,
      voiceEnabled: voice['enabled'] as bool? ?? false,
      voiceListening: voice['listening'] as bool? ?? false,
      voiceStatus: voice['status'] as String? ?? 'disabled',
      voiceTranscript: voice['transcript'] as String? ?? '',
      voiceReply: voice['reply'] as String? ?? '',
      voiceError: voice['error'] as String?,
      voiceInputDevice: voice['inputDevice'] as String? ?? 'pulse',
      voiceOutputDevice: voice['outputDevice'] as String? ?? 'pulse',
      voiceTtsEngine: voice['ttsEngine'] as String?,
      tools: _asList(
        json['tools'],
      ).map(_asMap).map(ToolCall.fromJson).toList(growable: false),
      logs: _asList(
        json['logs'],
      ).map(_asMap).map(ConsoleLog.fromJson).toList(growable: false),
    );
  }

  final bool backend;
  final bool starting;
  final bool busy;
  final bool promptSaved;
  final String systemPrompt;
  final String sessionId;
  final String cameraSource;
  final String modelStatus;
  final String skillStatus;
  final String skillName;
  final int progress;
  final String progressText;
  final int activeStep;
  final String currentTask;
  final String modelOutput;
  final double modelDurationSeconds;
  final int latency;
  final int taskCount;
  final String robotMode;
  final bool robotConnected;
  final String? robotModel;
  final bool? telemetryAvailable;
  final String cameraLabel;
  final String cameraStatus;
  final bool cameraFrameAvailable;
  final String cameraFramePath;
  final int cameraFrameVersion;
  final int cameraWidth;
  final int cameraHeight;
  final int cameraFps;
  final String? cameraError;
  final bool voiceEnabled;
  final bool voiceListening;
  final String voiceStatus;
  final String voiceTranscript;
  final String voiceReply;
  final String? voiceError;
  final String voiceInputDevice;
  final String voiceOutputDevice;
  final String? voiceTtsEngine;
  final List<ToolCall> tools;
  final List<ConsoleLog> logs;
}

Map<String, dynamic> _asMap(Object? value) {
  if (value is Map<String, dynamic>) return value;
  if (value is Map) {
    return value.map((key, item) => MapEntry(key.toString(), item));
  }
  return const {};
}

List<Object?> _asList(Object? value) => value is List ? value : const [];
