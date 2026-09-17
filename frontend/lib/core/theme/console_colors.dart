import 'package:flutter/material.dart';

abstract final class ConsoleColors {
  // Surfaces
  static const bg0 = Color(0xFF08090B);
  static const bg1 = Color(0xFF0D0F12);
  static const bg2 = Color(0xFF121519);
  static const panel = Color(0xFF14181E);
  static const panelElevated = Color(0xFF181C23);
  static const field = Color(0xFF0F1216);

  // Lines & ink
  static const line = Color(0xFF1E2430);
  static const lineSoft = Color(0xFF1A1F28);
  static const ink = Color(0xFFE8EDF5);
  static const muted = Color(0xFF8B96A8);
  static const dim = Color(0xFF5C6678);
  static const faint = Color(0xFF3A4250);

  // Status
  static const accent = Color(0xFF4FD1FF);
  static const blue = Color(0xFF3B82F6);
  static const green = Color(0xFF22C55E);
  static const amber = Color(0xFFF59E0B);
  static const red = Color(0xFFEF4444);
  static const purple = Color(0xFFA78BFA);

  // Legacy aliases used across panels
  static const cyan = accent;
}
