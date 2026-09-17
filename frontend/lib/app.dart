import 'package:flutter/material.dart';

import 'core/theme/app_theme.dart';
import 'features/console/presentation/console_page.dart';
import 'features/console/services/console_api.dart';

class G1ConsoleApp extends StatelessWidget {
  const G1ConsoleApp({super.key, this.api});

  final ConsoleApi? api;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Robotics Mission Control',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.dark,
      darkTheme: AppTheme.dark,
      themeMode: ThemeMode.dark,
      home: G1ConsolePage(api: api),
    );
  }
}
