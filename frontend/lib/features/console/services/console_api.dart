import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';

import '../models/console_snapshot.dart';
import 'console_transport.dart';

class ConsoleApiException implements Exception {
  const ConsoleApiException(this.message, {this.statusCode});

  final String message;
  final int? statusCode;

  @override
  String toString() => message;
}

abstract interface class ConsoleApi {
  Uri get baseUri;

  Future<ConsoleSnapshot> getConsole();
  Future<ConsoleSnapshot> startSession();
  Future<ConsoleSnapshot> stopSession();
  Future<ConsoleSnapshot> updateSystemPrompt(String prompt);
  Future<ConsoleSnapshot> submitTask(
    String instruction, {
    String? cameraSource,
    String? taskMode,
  });
  Future<ConsoleSnapshot> cancelTask(String reason);
  Future<ConsoleSnapshot> setCameraSource(String source);
  Future<ConsoleSnapshot> clearLogs();
  Future<ConsoleSnapshot> startVoice();
  Future<ConsoleSnapshot> stopVoice();
  Future<ConsoleSnapshot> speak(String text);
  Future<Uint8List> fetchCameraFrame(String path);
  Stream<Map<String, dynamic>> events();
  Uri cameraFrameUri(String path, int version);
  Future<void> close();
}

class HttpConsoleApi implements ConsoleApi {
  HttpConsoleApi({Uri? baseUri, http.Client? client})
    : this._(baseUri ?? Uri.parse(defaultBaseUrl), client);

  HttpConsoleApi._(this.baseUri, http.Client? client)
    : _client = client ?? createConsoleHttpClient(baseUri);

  static const defaultBaseUrl = String.fromEnvironment(
    'G1_API_BASE_URL',
    defaultValue: 'http://192.168.31.74:8000',
  );

  @override
  final Uri baseUri;
  final http.Client _client;
  WebSocketChannel? _eventChannel;

  Uri _uri(String path) =>
      baseUri.replace(path: path, query: null, fragment: null);

  @override
  Future<ConsoleSnapshot> getConsole() async =>
      ConsoleSnapshot.fromJson(await _get('/api/v1/console'));

  @override
  Future<ConsoleSnapshot> startSession() async =>
      ConsoleSnapshot.fromJson(await _post('/api/v1/session/start'));

  @override
  Future<ConsoleSnapshot> stopSession() async =>
      ConsoleSnapshot.fromJson(await _post('/api/v1/session/stop'));

  @override
  Future<ConsoleSnapshot> updateSystemPrompt(String prompt) async =>
      ConsoleSnapshot.fromJson(
        await _put('/api/v1/config/system-prompt', {'systemPrompt': prompt}),
      );

  @override
  Future<ConsoleSnapshot> submitTask(
    String instruction, {
    String? cameraSource,
    String? taskMode,
  }) async => ConsoleSnapshot.fromJson(
    await _post('/api/v1/tasks', {
      'instruction': instruction,
      'cameraSource': cameraSource,
      'taskMode': taskMode,
    }),
  );

  @override
  Future<ConsoleSnapshot> cancelTask(String reason) async =>
      ConsoleSnapshot.fromJson(
        await _post('/api/v1/tasks/current/cancel', {'reason': reason}),
      );

  @override
  Future<ConsoleSnapshot> setCameraSource(String source) async =>
      ConsoleSnapshot.fromJson(
        await _put('/api/v1/camera/source', {'source': source}),
      );

  @override
  Future<ConsoleSnapshot> clearLogs() async =>
      ConsoleSnapshot.fromJson(await _delete('/api/v1/logs'));

  @override
  Future<ConsoleSnapshot> startVoice() async =>
      ConsoleSnapshot.fromJson(await _post('/api/v1/voice/start'));

  @override
  Future<ConsoleSnapshot> stopVoice() async =>
      ConsoleSnapshot.fromJson(await _post('/api/v1/voice/stop'));

  @override
  Future<ConsoleSnapshot> speak(String text) async => ConsoleSnapshot.fromJson(
    await _post('/api/v1/voice/speak', {'text': text}),
  );

  @override
  Future<Uint8List> fetchCameraFrame(String path) async {
    final uri = _uri(path).replace(
      queryParameters: {'t': DateTime.now().microsecondsSinceEpoch.toString()},
    );
    final response = await _client.get(uri);
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw ConsoleApiException(
        '摄像头画面请求失败（HTTP ${response.statusCode}）',
        statusCode: response.statusCode,
      );
    }
    return response.bodyBytes;
  }

  @override
  Stream<Map<String, dynamic>> events() async* {
    final eventUri = baseUri.replace(
      scheme: baseUri.scheme == 'https' ? 'wss' : 'ws',
      path: '/api/v1/events',
      query: null,
      fragment: null,
    );
    final channel = connectConsoleWebSocket(eventUri);
    _eventChannel = channel;
    try {
      await channel.ready;
      await for (final raw in channel.stream) {
        final decoded = jsonDecode(raw as String);
        if (decoded is Map<String, dynamic>) yield decoded;
      }
    } finally {
      if (identical(_eventChannel, channel)) _eventChannel = null;
      await channel.sink.close();
    }
  }

  @override
  Uri cameraFrameUri(String path, int version) {
    final resolved = baseUri.resolve(path);
    return resolved.replace(queryParameters: {'v': '$version'});
  }

  Future<Map<String, dynamic>> _get(String path) async =>
      _decode(await _client.get(_uri(path)));

  Future<Map<String, dynamic>> _post(
    String path, [
    Map<String, Object?>? body,
  ]) async => _decode(
    await _client.post(
      _uri(path),
      headers: const {'content-type': 'application/json'},
      body: body == null ? null : jsonEncode(body),
    ),
  );

  Future<Map<String, dynamic>> _put(
    String path,
    Map<String, Object?> body,
  ) async => _decode(
    await _client.put(
      _uri(path),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode(body),
    ),
  );

  Future<Map<String, dynamic>> _delete(String path) async =>
      _decode(await _client.delete(_uri(path)));

  Map<String, dynamic> _decode(http.Response response) {
    Object? decoded;
    try {
      decoded = jsonDecode(utf8.decode(response.bodyBytes));
    } on FormatException {
      decoded = null;
    }
    if (response.statusCode < 200 || response.statusCode >= 300) {
      final detail = decoded is Map ? decoded['detail'] : null;
      throw ConsoleApiException(
        detail?.toString() ?? '后端请求失败（HTTP ${response.statusCode}）',
        statusCode: response.statusCode,
      );
    }
    if (decoded is! Map) {
      throw const ConsoleApiException('后端返回了无效的 JSON 对象');
    }
    return decoded.map((key, value) => MapEntry(key.toString(), value));
  }

  @override
  Future<void> close() async {
    final channel = _eventChannel;
    _eventChannel = null;
    if (channel != null) await channel.sink.close();
    _client.close();
  }
}
