import 'dart:async';

import 'package:flutter/material.dart';

import '../models/console_log.dart';
import '../models/console_snapshot.dart';
import '../models/tool_call.dart';
import '../services/console_api.dart';

class ConsoleController extends ChangeNotifier {
  ConsoleController({required this.onMessage, ConsoleApi? api})
    : api = api ?? HttpConsoleApi();

  final void Function(String message) onMessage;
  final ConsoleApi api;

  final systemPromptController = TextEditingController(
    text:
        '你是 G1 机器人的任务助手。\n先观察摄像头画面，再规划任务。\n调用工具与技能时，输出执行状态。\n遇到障碍或不确定情况时，停止并报告。',
  );
  final taskController = TextEditingController();
  final searchController = TextEditingController();
  final logScrollController = ScrollController();

  final List<ConsoleLog> logs = [];
  final List<ToolCall> tools = [];

  Timer? clockTimer;
  Timer? _pollTimer;
  Timer? _reconnectTimer;
  StreamSubscription<Map<String, dynamic>>? _eventSubscription;
  DateTime now = DateTime.now();
  DateTime? taskStartedAt;
  bool backend = false;
  bool starting = false;
  bool busy = false;
  bool promptSaved = true;
  bool cameraExpanded = false;
  bool robotConnected = false;
  bool connectedToApi = false;
  int taskCount = 0;
  int latency = 0;
  int progress = 0;
  int activeStep = -1;
  int cameraFrameVersion = 0;
  int cameraWidth = 640;
  int cameraHeight = 480;
  int cameraFps = 30;
  String sessionId = '—';
  String cameraSource = 'demo';
  String taskMode = 'text';
  String waveResponse = 'wave';
  void setWaveResponse(String value) {
    if (busy || !['wave', 'heart'].contains(value)) return;
    waveResponse = value;
    refresh();
  }

  bool get gestureMode => taskMode == 'gesture';

  void setTaskMode(String mode) {
    if (busy || (mode != 'text' && mode != 'gesture')) return;
    taskMode = mode;
    refresh();
  }

  String cameraLabel = '模拟视频源';
  String cameraStatus = 'idle';
  String cameraFramePath = '/api/v1/camera/frame.jpg';
  String? cameraError;
  bool cameraFrameAvailable = false;
  String robotMode = 'simulation';
  String? robotModel;
  bool? telemetryAvailable;
  String logLevel = 'ALL';
  String modelStatus = '待命';
  String skillStatus = 'IDLE';
  String skillName = '等待调度';
  String progressText = '等待执行';
  String modelOutput = '';
  String currentTask = '';
  Duration modelDuration = Duration.zero;

  bool _initialized = false;
  bool _disposed = false;
  bool _applyingServerPrompt = false;
  bool _refreshInFlight = false;
  bool _logScrollScheduled = false;

  bool get isActive => !_disposed;
  bool get isHardware => robotMode == 'hardware';
  String get robotModelLabel => robotModel ?? 'ROBOT';
  bool get telemetryUnavailable => telemetryAvailable == false;
  String get apiBaseUrl => api.baseUri.toString();
  String get cameraFrameUrl =>
      api.cameraFrameUri(cameraFramePath, cameraFrameVersion).toString();

  void initialize() {
    if (!isActive || _initialized) return;
    _initialized = true;
    systemPromptController.addListener(onPromptChanged);
    taskController.addListener(refresh);
    searchController.addListener(_onLogFilterChanged);
    addLog('INFO', 'console', '正在连接 G1 FastAPI 后端。', refresh: false);
    clockTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!isActive) return;
      now = DateTime.now();
      if (busy && taskStartedAt != null) {
        modelDuration = now.difference(taskStartedAt!);
      }
      _update(() {});
    });
    _pollTimer = Timer.periodic(
      const Duration(seconds: 2),
      (_) => unawaited(refreshFromBackend(reportError: false)),
    );
    unawaited(_bootstrap());
  }

  Future<void> _bootstrap() async {
    try {
      await refreshFromBackend(reportError: true);
    } finally {
      if (isActive) _connectEvents();
    }
  }

  Future<void> refreshFromBackend({bool reportError = true}) async {
    if (!isActive || _refreshInFlight) return;
    _refreshInFlight = true;
    try {
      final snapshot = await api.getConsole();
      connectedToApi = true;
      _applySnapshot(snapshot);
    } catch (error) {
      connectedToApi = false;
      if (reportError) {
        addLog('ERROR', 'network', '无法连接 FastAPI：$error');
        onMessage('无法连接后端：$error');
      } else {
        _update(() {});
      }
    } finally {
      _refreshInFlight = false;
    }
  }

  void _connectEvents() {
    if (!isActive || _eventSubscription != null) return;
    _eventSubscription = api.events().listen(
      _handleEvent,
      onError: (Object error, StackTrace stackTrace) {
        connectedToApi = false;
        _eventSubscription = null;
        if (isActive) {
          addLog('WARN', 'network', '实时事件连接中断：$error');
          _scheduleReconnect();
        }
      },
      onDone: () {
        connectedToApi = false;
        _eventSubscription = null;
        if (isActive) _scheduleReconnect();
      },
    );
  }

  void _scheduleReconnect() {
    if (_reconnectTimer?.isActive ?? false) return;
    _reconnectTimer = Timer(const Duration(seconds: 2), () {
      if (!isActive) return;
      unawaited(refreshFromBackend(reportError: false));
      _connectEvents();
    });
  }

  void _handleEvent(Map<String, dynamic> event) {
    if (!isActive) return;
    connectedToApi = true;
    final data = _asMap(event['data']);
    switch (event['type']) {
      case 'state':
        _applySnapshot(ConsoleSnapshot.fromJson(data));
      case 'log':
        _mergeLog(ConsoleLog.fromJson(data));
        _update(() {});
      case 'heartbeat':
        latency = (data['latencyMs'] as num?)?.toInt() ?? latency;
        robotConnected = data['connected'] as bool? ?? robotConnected;
        _update(() {});
      case 'camera':
        final shouldRefresh = !cameraFrameAvailable || cameraStatus != 'ready';
        cameraFrameVersion =
            (data['frameVersion'] as num?)?.toInt() ?? cameraFrameVersion;
        cameraFrameAvailable = true;
        cameraStatus = 'ready';
        // The image widget pulls frames independently. Rebuilding the whole
        // console for every camera frame starves image decoding on Flutter Web.
        if (shouldRefresh) _update(() {});
    }
  }

  void _applySnapshot(ConsoleSnapshot snapshot, {bool forcePrompt = false}) {
    if (!isActive) return;
    final wasBusy = busy;
    final promptIsDirty = !promptSaved && !forcePrompt;
    backend = snapshot.backend;
    starting = snapshot.starting;
    busy = snapshot.busy;
    promptSaved = promptIsDirty ? false : snapshot.promptSaved;
    sessionId = snapshot.sessionId;
    cameraSource = snapshot.cameraSource;
    modelStatus = snapshot.modelStatus;
    skillStatus = snapshot.skillStatus;
    skillName = snapshot.skillName;
    progress = snapshot.progress;
    progressText = snapshot.progressText;
    activeStep = snapshot.activeStep;
    currentTask = snapshot.currentTask;
    modelOutput = snapshot.modelOutput;
    taskCount = snapshot.taskCount;
    latency = snapshot.latency;
    robotMode = snapshot.robotMode;
    robotModel = snapshot.robotModel;
    telemetryAvailable = snapshot.telemetryAvailable;
    robotConnected = snapshot.robotConnected;
    cameraLabel = snapshot.cameraLabel;
    cameraStatus = snapshot.cameraStatus;
    cameraFrameAvailable = snapshot.cameraFrameAvailable;
    cameraFramePath = snapshot.cameraFramePath;
    cameraFrameVersion = snapshot.cameraFrameVersion;
    cameraWidth = snapshot.cameraWidth;
    cameraHeight = snapshot.cameraHeight;
    cameraFps = snapshot.cameraFps;
    cameraError = snapshot.cameraError;

    if (!promptIsDirty &&
        systemPromptController.text != snapshot.systemPrompt) {
      _applyingServerPrompt = true;
      systemPromptController.text = snapshot.systemPrompt;
      _applyingServerPrompt = false;
    }

    if (busy) {
      if (!wasBusy || taskStartedAt == null) taskStartedAt = DateTime.now();
      final serverDuration = Duration(
        milliseconds: (snapshot.modelDurationSeconds * 1000).round(),
      );
      if (serverDuration > modelDuration) modelDuration = serverDuration;
    } else {
      taskStartedAt = null;
      modelDuration = Duration(
        milliseconds: (snapshot.modelDurationSeconds * 1000).round(),
      );
    }

    tools
      ..clear()
      ..addAll(snapshot.tools);
    logs
      ..clear()
      ..addAll(snapshot.logs.take(500));
    notifyListeners();
    _scheduleLogScroll();
  }

  @override
  void dispose() {
    _disposed = true;
    clockTimer?.cancel();
    _pollTimer?.cancel();
    _reconnectTimer?.cancel();
    unawaited(_eventSubscription?.cancel());
    _eventSubscription = null;
    unawaited(api.close());
    systemPromptController.dispose();
    taskController.dispose();
    searchController.dispose();
    logScrollController.dispose();
    super.dispose();
  }

  void refresh() {
    if (isActive) _update(() {});
  }

  void _onLogFilterChanged() {
    if (!isActive) return;
    notifyListeners();
    _scheduleLogScroll();
  }

  void onPromptChanged() {
    if (!_applyingServerPrompt && promptSaved) {
      _update(() => promptSaved = false);
    }
  }

  String formattedTime([DateTime? value]) {
    final t = value ?? now;
    String two(int number) => number.toString().padLeft(2, '0');
    return '${two(t.hour)}:${two(t.minute)}:${two(t.second)}';
  }

  void addLog(
    String level,
    String source,
    String message, {
    bool refresh = true,
  }) {
    _mergeLog(
      ConsoleLog(formattedTime(DateTime.now()), level, source, message),
    );
    if (refresh && isActive) _update(() {});
  }

  void _mergeLog(ConsoleLog entry) {
    if (entry.id.isNotEmpty && logs.any((item) => item.id == entry.id)) return;
    logs.add(entry);
    if (logs.length > 500) logs.removeRange(0, logs.length - 500);
    _scheduleLogScroll();
  }

  void _scheduleLogScroll() {
    if (!isActive || _logScrollScheduled) return;
    _logScrollScheduled = true;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _logScrollScheduled = false;
      if (!isActive || !logScrollController.hasClients) return;
      final position = logScrollController.position;
      if (position.pixels == position.maxScrollExtent) return;
      // New events can arrive several times per frame. Jumping after layout is
      // deterministic and cannot be interrupted by another scroll animation.
      logScrollController.jumpTo(position.maxScrollExtent);
    });
  }

  Future<void> toggleBackend() async {
    if (starting) return;
    _update(() => starting = true);
    try {
      final snapshot = backend
          ? await api.stopSession()
          : await api.startSession();
      connectedToApi = true;
      _applySnapshot(snapshot);
      onMessage(backend ? '后端已启动' : '后端已停止');
    } catch (error) {
      starting = false;
      addLog('ERROR', 'network', '后端状态切换失败：$error');
      onMessage('操作失败：$error');
    }
  }

  Future<void> savePrompt() async {
    final prompt = systemPromptController.text.trim();
    if (prompt.isEmpty) {
      onMessage('系统提示词不能为空');
      return;
    }
    try {
      final snapshot = await api.updateSystemPrompt(prompt);
      _applySnapshot(snapshot, forcePrompt: true);
      onMessage('系统提示词已保存');
    } catch (error) {
      addLog('ERROR', 'config', '系统提示词保存失败：$error');
      onMessage('保存失败：$error');
    }
  }

  Future<void> submitTask() async {
    final prompt = taskController.text.trim();
    if (!backend || busy || prompt.isEmpty) return;
    if (!promptSaved) {
      onMessage('请先保存修改后的系统提示词');
      return;
    }
    if (gestureMode && cameraSource != 'local') {
      onMessage('手势交互需要先选择本地相机');
      return;
    }
    try {
      final snapshot = await api.submitTask(
        prompt,
        cameraSource: cameraSource,
        taskMode: taskMode,
        waveResponse: gestureMode && robotModel == 'GO2'
            ? waveResponse
            : 'wave',
      );
      _applySnapshot(snapshot);
    } catch (error) {
      addLog('ERROR', 'agent', '任务提交失败：$error');
      onMessage('任务提交失败：$error');
    }
  }

  Future<void> cancelTask([String reason = '用户停止了任务']) async {
    if (!busy) return;
    try {
      final snapshot = await api.cancelTask(reason);
      _applySnapshot(snapshot);
      onMessage('任务已停止');
    } catch (error) {
      addLog('ERROR', 'executor', '任务停止失败：$error');
      onMessage('停止失败：$error');
    }
  }

  List<ConsoleLog> get visibleLogs {
    final query = searchController.text.trim().toLowerCase();
    return logs.where((entry) {
      final levelMatches = logLevel == 'ALL' || entry.level == logLevel;
      final queryMatches =
          query.isEmpty ||
          '${entry.source} ${entry.message}'.toLowerCase().contains(query);
      return levelMatches && queryMatches;
    }).toList();
  }

  void _update(VoidCallback change) {
    if (!isActive) return;
    change();
    notifyListeners();
  }

  Future<void> selectCameraSource(String value) async {
    if (value == cameraSource) return;
    try {
      final snapshot = await api.setCameraSource(value);
      _applySnapshot(snapshot);
      onMessage(value == 'local' ? '已切换至 D435i' : '已切换至模拟视频源');
    } catch (error) {
      addLog('ERROR', 'camera', '摄像头切换失败：$error');
      onMessage('摄像头切换失败：$error');
    }
  }

  void toggleCameraExpanded() =>
      _update(() => cameraExpanded = !cameraExpanded);
  void setLogLevel(String value) {
    _update(() => logLevel = value);
    _scheduleLogScroll();
  }

  Future<void> clearLogs() async {
    try {
      final snapshot = await api.clearLogs();
      _applySnapshot(snapshot);
      onMessage('日志已清空');
    } catch (error) {
      addLog('ERROR', 'network', '清空日志失败：$error');
      onMessage('清空失败：$error');
    }
  }
}

Map<String, dynamic> _asMap(Object? value) {
  if (value is Map<String, dynamic>) return value;
  if (value is Map) {
    return value.map((key, item) => MapEntry(key.toString(), item));
  }
  return const {};
}
