import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../../../core/theme/console_colors.dart';
import '../controllers/console_controller.dart';
import '../services/console_api.dart';
import 'console_intents.dart';
import 'widgets/camera_panel.dart';
import 'widgets/model_panel.dart';
import 'widgets/tool_panel.dart';
import 'widgets/skill_panel.dart';
import 'widgets/robot_status_panel.dart';
import 'widgets/prompt_panel.dart';
import 'widgets/backend_panel.dart';
import 'widgets/voice_panel.dart';
import 'widgets/task_input_panel.dart';
import 'widgets/stop_task_button.dart';
import 'widgets/log_panel.dart';
import 'widgets/console_rail.dart';
import 'widgets/console_header.dart';
import 'widgets/console_heading.dart';
import 'widgets/console_footer.dart';

class G1ConsolePage extends StatefulWidget {
  const G1ConsolePage({super.key, this.api});

  final ConsoleApi? api;

  @override
  State<G1ConsolePage> createState() => _G1ConsolePageState();
}

class _G1ConsolePageState extends State<G1ConsolePage> {
  late final ConsoleController controller;
  final anchors = ConsoleAnchors();

  @override
  void initState() {
    super.initState();
    controller = ConsoleController(onMessage: _toast, api: widget.api)
      ..initialize();
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  void _toast(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          content: Text(message),
          behavior: SnackBarBehavior.floating,
          width: min(MediaQuery.sizeOf(context).width - 32, 420),
          backgroundColor: const Color(0xFF1A2030),
          duration: const Duration(seconds: 2),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
            side: const BorderSide(color: ConsoleColors.line),
          ),
        ),
      );
  }

  void _scrollToAnchor(GlobalKey key) {
    final target = key.currentContext;
    if (target == null) return;
    Scrollable.ensureVisible(
      target,
      duration: const Duration(milliseconds: 220),
      curve: Curves.easeOut,
      alignment: 0.04,
    );
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: controller,
      builder: (context, child) => Shortcuts(
        shortcuts: const {
          SingleActivator(LogicalKeyboardKey.enter, control: true):
              SubmitTaskIntent(),
          SingleActivator(LogicalKeyboardKey.enter, meta: true):
              SubmitTaskIntent(),
          SingleActivator(LogicalKeyboardKey.escape): StopTaskIntent(),
        },
        child: Actions(
          actions: {
            SubmitTaskIntent: CallbackAction<SubmitTaskIntent>(
              onInvoke: (_) {
                controller.submitTask();
                return null;
              },
            ),
            StopTaskIntent: CallbackAction<StopTaskIntent>(
              onInvoke: (_) {
                controller.cancelTask();
                return null;
              },
            ),
          },
          child: Focus(
            autofocus: true,
            child: Scaffold(
              backgroundColor: ConsoleColors.bg0,
              // Keeps the status HUD clear of the iOS status bar / dynamic
              // island and the Android system bars. No-op on desktop.
              body: SafeArea(
                bottom: false,
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    final showRail = constraints.maxWidth > 640;
                    final double railWidth = constraints.maxWidth > 930
                        ? 76
                        : 64;
                    return Row(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        if (showRail)
                          ConsoleRail(
                            width: railWidth,
                            anchors: anchors,
                            onNavigate: _scrollToAnchor,
                          ),
                        Expanded(
                          child: _buildApplication(
                            constraints.maxWidth - (showRail ? railWidth : 0),
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildApplication(double width) {
    return Column(
      children: [
        ConsoleHeader(controller: controller, width: width),
        Expanded(
          child: SingleChildScrollView(
            padding: EdgeInsets.fromLTRB(
              width <= 640 ? 12 : 24,
              width <= 640 ? 14 : 20,
              width <= 640 ? 12 : 24,
              0,
            ),
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1680),
                child: Column(
                  children: [
                    ConsoleHeading(controller: controller, width: width),
                    const SizedBox(height: 18),
                    _buildWorkspace(width),
                    const SizedBox(height: 16),
                    LogPanel(controller: controller, width: width),
                    ConsoleFooter(controller: controller, width: width),
                  ],
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildWorkspace(double width) {
    final twoColumns = width > 930;
    final monitor = Column(
      children: [
        CameraPanel(controller: controller, anchorKey: anchors.mission),
        const SizedBox(height: 14),
        _buildExecutionPanels(),
      ],
    );
    final controls = _buildControls(width);
    if (twoColumns) {
      return Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(child: monitor),
          const SizedBox(width: 16),
          SizedBox(width: width >= 1500 ? 340 : 310, child: controls),
        ],
      );
    }
    return Column(children: [monitor, const SizedBox(height: 14), controls]);
  }

  Widget _buildExecutionPanels() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final panels = <Widget>[
          ModelPanel(controller: controller),
          ToolPanel(controller: controller),
          SkillPanel(controller: controller),
        ];
        if (constraints.maxWidth >= 900) {
          return Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              for (var i = 0; i < panels.length; i++) ...[
                Expanded(flex: i == 2 ? 106 : 100, child: panels[i]),
                if (i < panels.length - 1) const SizedBox(width: 12),
              ],
            ],
          );
        }
        if (constraints.maxWidth >= 620) {
          return Column(
            children: [
              Row(
                children: [
                  Expanded(child: panels[0]),
                  const SizedBox(width: 12),
                  Expanded(child: panels[1]),
                ],
              ),
              const SizedBox(height: 12),
              panels[2],
            ],
          );
        }
        return Column(
          children: [
            panels[0],
            const SizedBox(height: 12),
            panels[1],
            const SizedBox(height: 12),
            panels[2],
          ],
        );
      },
    );
  }

  Widget _buildControls(double width) {
    final items = <Widget>[
      RobotStatusPanel(controller: controller, anchorKey: anchors.robot),
      TaskInputPanel(controller: controller),
      StopTaskButton(controller: controller),
      BackendPanel(controller: controller, anchorKey: anchors.system),
      VoicePanel(controller: controller),
      PromptPanel(controller: controller),
    ];
    if (width <= 930 && width > 640) {
      return Column(
        children: [
          items[0],
          const SizedBox(height: 12),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(child: items[1]),
              const SizedBox(width: 12),
              Expanded(child: items[2]),
            ],
          ),
          const SizedBox(height: 12),
          items[3],
          const SizedBox(height: 12),
          items[4],
        ],
      );
    }
    return Column(
      children: [
        for (var i = 0; i < items.length; i++) ...[
          items[i],
          if (i < items.length - 1) const SizedBox(height: 12),
        ],
      ],
    );
  }
}
