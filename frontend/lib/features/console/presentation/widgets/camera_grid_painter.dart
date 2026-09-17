import 'dart:math';

import 'package:flutter/material.dart';

class CameraGridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final background = Paint()
      ..shader =
          RadialGradient(
            colors: const [Color(0xFF12171E), Color(0xFF0A0C10)],
            stops: const [0, 1],
            radius: .8,
          ).createShader(
            Rect.fromCircle(center: center, radius: size.longestSide * .7),
          );
    canvas.drawRect(Offset.zero & size, background);

    final gridPaint = Paint()
      ..color = const Color(0xFF8AA0C0).withValues(alpha: .045)
      ..strokeWidth = 1;
    const gap = 36.0;
    for (double x = (size.width % gap) / 2; x < size.width; x += gap) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), gridPaint);
    }
    for (double y = (size.height % gap) / 2; y < size.height; y += gap) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    final guide = Paint()
      ..color = const Color(0xFF4FD1FF).withValues(alpha: .18)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    final boxWidth = min(size.width * .58, 420.0);
    final boxHeight = size.height * .62;
    const corner = 16.0;
    final rect = Rect.fromCenter(
      center: center,
      width: boxWidth,
      height: boxHeight,
    );
    final path = Path()
      ..moveTo(rect.left + corner, rect.top)
      ..lineTo(rect.left, rect.top)
      ..lineTo(rect.left, rect.top + corner)
      ..moveTo(rect.right - corner, rect.top)
      ..lineTo(rect.right, rect.top)
      ..lineTo(rect.right, rect.top + corner)
      ..moveTo(rect.left, rect.bottom - corner)
      ..lineTo(rect.left, rect.bottom)
      ..lineTo(rect.left + corner, rect.bottom)
      ..moveTo(rect.right - corner, rect.bottom)
      ..lineTo(rect.right, rect.bottom)
      ..lineTo(rect.right, rect.bottom - corner);
    canvas.drawPath(path, guide);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
