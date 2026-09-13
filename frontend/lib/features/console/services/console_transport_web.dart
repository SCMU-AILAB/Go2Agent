import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';

http.Client createConsoleHttpClient(Uri baseUri) => http.Client();

WebSocketChannel connectConsoleWebSocket(Uri uri) =>
    WebSocketChannel.connect(uri);
