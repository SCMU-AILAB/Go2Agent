import 'package:flutter/material.dart';

import 'core/theme/app_theme.dart';
import 'features/console/presentation/console_page.dart';
import 'features/console/services/console_api.dart';

class Go2ConsoleApp extends StatelessWidget {
  const Go2ConsoleApp({super.key, this.api});

  final ConsoleApi? api;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Robotics Mission Control',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.dark,
      darkTheme: AppTheme.dark,
      themeMode: ThemeMode.dark,
      home: Go2ConsolePage(api: api),
    );
  }
}
