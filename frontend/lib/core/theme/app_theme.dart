import 'package:flutter/material.dart';

import 'console_colors.dart';

abstract final class AppTheme {
  static final dark = ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    scaffoldBackgroundColor: ConsoleColors.bg0,
    colorScheme: const ColorScheme.dark(
      primary: ConsoleColors.accent,
      onPrimary: ConsoleColors.bg0,
      secondary: ConsoleColors.blue,
      surface: ConsoleColors.panel,
      onSurface: ConsoleColors.ink,
      error: ConsoleColors.red,
      outline: ConsoleColors.line,
    ),
    fontFamily: 'PingFang SC',
    dividerColor: ConsoleColors.line,
    canvasColor: ConsoleColors.bg1,
    textTheme: const TextTheme(
      bodyLarge: TextStyle(color: ConsoleColors.ink, fontSize: 14, height: 1.5),
      bodyMedium: TextStyle(
        color: ConsoleColors.ink,
        fontSize: 13,
        height: 1.5,
      ),
      bodySmall: TextStyle(color: ConsoleColors.muted, fontSize: 12),
      titleSmall: TextStyle(
        color: ConsoleColors.ink,
        fontSize: 12,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.8,
      ),
      labelSmall: TextStyle(
        color: ConsoleColors.dim,
        fontSize: 10,
        letterSpacing: 1.2,
        fontWeight: FontWeight.w600,
      ),
      // Buttons and dropdowns otherwise fall back to the platform font, which
      // breaks the typography system on mixed CJK text.
      labelLarge: TextStyle(
        color: ConsoleColors.ink,
        fontSize: 13,
        fontWeight: FontWeight.w600,
        fontFamily: 'PingFang SC',
      ),
      titleMedium: TextStyle(
        color: ConsoleColors.ink,
        fontSize: 14,
        fontWeight: FontWeight.w600,
        fontFamily: 'PingFang SC',
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: ConsoleColors.field,
      contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 11),
      hintStyle: const TextStyle(color: ConsoleColors.dim, fontSize: 13),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(6),
        borderSide: const BorderSide(color: ConsoleColors.line),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(6),
        borderSide: const BorderSide(color: ConsoleColors.accent, width: 1),
      ),
    ),
    scrollbarTheme: ScrollbarThemeData(
      thumbColor: WidgetStatePropertyAll(
        ConsoleColors.faint.withValues(alpha: .5),
      ),
    ),
    snackBarTheme: const SnackBarThemeData(
      backgroundColor: Color(0xFF1A2030),
      contentTextStyle: TextStyle(color: ConsoleColors.ink, fontSize: 13),
      behavior: SnackBarBehavior.floating,
    ),
  );

  // Keep name used by older imports / tests.
  static final light = dark;
}
