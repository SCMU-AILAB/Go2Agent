import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:http/io_client.dart';
import 'package:web_socket_channel/io.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

http.Client createConsoleHttpClient(Uri baseUri) =>
    IOClient(_createHttpClient(baseUri));

WebSocketChannel connectConsoleWebSocket(Uri uri) =>
    IOWebSocketChannel.connect(uri, customClient: _createHttpClient(uri));

HttpClient _createHttpClient(Uri target) {
  final client = HttpClient();
  if (_isPrivateHost(target.host)) {
    // macOS can otherwise send robot-LAN traffic through the system proxy.
    client.findProxy = (_) => 'DIRECT';
  }
  return client;
}

bool _isPrivateHost(String host) {
  final normalized = host.toLowerCase();
  if (normalized == 'localhost' || normalized == '::1') return true;

  final address = InternetAddress.tryParse(normalized);
  if (address == null) return false;
  if (address.isLoopback || address.isLinkLocal) return true;
  if (address.type != InternetAddressType.IPv4) return false;

  final octets = address.rawAddress;
  return octets[0] == 10 ||
      (octets[0] == 172 && octets[1] >= 16 && octets[1] <= 31) ||
      (octets[0] == 192 && octets[1] == 168);
}
