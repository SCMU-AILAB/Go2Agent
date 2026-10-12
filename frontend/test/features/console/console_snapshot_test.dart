import 'package:flutter_test/flutter_test.dart';
import 'package:go2_frontend/features/console/models/console_snapshot.dart';

import '../../support/fake_console_api.dart';

void main() {
  test('reads the backend visionConfirmHoldS field', () {
    final snapshot = ConsoleSnapshot.fromJson({
      ...consolePayload(),
      'visionConfirmHoldS': 2.25,
    });

    expect(snapshot.visionConfirmHoldS, 2.25);
  });

  test('defaults confirm hold to the backend default', () {
    final payload = consolePayload()..remove('visionConfirmHoldS');

    expect(ConsoleSnapshot.fromJson(payload).visionConfirmHoldS, 0.0);
  });
}
