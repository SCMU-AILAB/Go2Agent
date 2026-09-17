import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'camera_grid_painter.dart';
import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';
import '../../services/console_api.dart';
import 'console_widgets.dart';

class CameraPanel extends StatelessWidget {
  const CameraPanel({super.key, required this.controller, this.anchorKey});

  final ConsoleController controller;
  final Key? anchorKey;

  @override
  Widget build(BuildContext context) {
    final liveCamera =
        controller.cameraSource == 'local' &&
        controller.cameraFrameAvailable &&
        controller.cameraStatus == 'ready';
    return Container(
      key: anchorKey,
      child: panel(
        header: sectionTitle(Icons.videocam_outlined, 'VISION FEED'),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              height: 30,
              padding: const EdgeInsets.symmetric(horizontal: 6),
              decoration: BoxDecoration(
                border: Border.all(color: ConsoleColors.line),
                borderRadius: BorderRadius.circular(5),
                color: ConsoleColors.field,
              ),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<String>(
                  value: controller.cameraSource,
                  dropdownColor: ConsoleColors.panelElevated,
                  borderRadius: BorderRadius.circular(6),
                  style: const TextStyle(
                    color: ConsoleColors.muted,
                    fontSize: 11,
                    fontFamily: 'PingFang SC',
                  ),
                  iconEnabledColor: ConsoleColors.muted,
                  items: const [
                    DropdownMenuItem(value: 'demo', child: Text('模拟视频源')),
                    DropdownMenuItem(value: 'local', child: Text('本机摄像头')),
                  ],
                  onChanged: (value) {
                    if (value == null) return;
                    controller.selectCameraSource(value);
                  },
                ),
              ),
            ),
          ],
        ),
        child: Column(
          children: [
            LayoutBuilder(
              builder: (context, constraints) {
                final normalHeight = (constraints.maxWidth * 0.5).clamp(
                  260.0,
                  480.0,
                );
                return AnimatedContainer(
                  duration: const Duration(milliseconds: 220),
                  curve: Curves.easeOut,
                  height: controller.cameraExpanded
                      ? min(MediaQuery.sizeOf(context).height * .72, 700)
                      : normalHeight,
                  margin: const EdgeInsets.fromLTRB(10, 10, 10, 0),
                  clipBehavior: Clip.antiAlias,
                  decoration: BoxDecoration(
                    color: const Color(0xFF0A0C10),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: ConsoleColors.lineSoft),
                  ),
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      if (liveCamera)
                        _LiveCameraFrame(
                          api: controller.api,
                          framePath: controller.cameraFramePath,
                          targetFps: controller.cameraFps,
                        ),
                      if (!liveCamera)
                        CustomPaint(painter: CameraGridPainter()),
                      if (!liveCamera)
                        Center(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Container(
                                width: 48,
                                height: 48,
                                alignment: Alignment.center,
                                decoration: BoxDecoration(
                                  color: ConsoleColors.accent.withValues(
                                    alpha: .04,
                                  ),
                                  border: Border.all(
                                    color: ConsoleColors.accent.withValues(
                                      alpha: .18,
                                    ),
                                  ),
                                  borderRadius: BorderRadius.circular(10),
                                ),
                                child: Icon(
                                  controller.cameraStatus == 'error'
                                      ? Icons.videocam_off_outlined
                                      : Icons.videocam_outlined,
                                  size: 24,
                                  color: ConsoleColors.muted,
                                ),
                              ),
                              const SizedBox(height: 14),
                              Text(
                                controller.cameraStatus == 'error'
                                    ? '视觉通道异常'
                                    : controller.cameraSource == 'local'
                                    ? '等待 D435i 画面'
                                    : '模拟视觉通道就绪',
                                style: const TextStyle(
                                  color: ConsoleColors.ink,
                                  fontSize: 14,
                                  fontWeight: FontWeight.w500,
                                  letterSpacing: 0.8,
                                ),
                              ),
                              const SizedBox(height: 6),
                              Padding(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 28,
                                ),
                                child: Text(
                                  controller.cameraError ??
                                      (controller.cameraSource == 'local'
                                          ? '画面由 FastAPI 后端的 USB RealSense 提供'
                                          : '切换到本机摄像头以启用 D435i'),
                                  textAlign: TextAlign.center,
                                  style: const TextStyle(
                                    color: ConsoleColors.dim,
                                    fontSize: 11,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      // HUD corners
                      Positioned(
                        left: 14,
                        top: 12,
                        child: darkChip(
                          controller.cameraSource == 'local'
                              ? 'D435i · ${controller.cameraStatus.toUpperCase()}'
                              : 'SIMULATED FEED',
                        ),
                      ),
                      Positioned(
                        right: 14,
                        top: 12,
                        child: darkChip(
                          'RGB  ${controller.cameraWidth}×${controller.cameraHeight}',
                        ),
                      ),
                      Positioned(
                        left: 14,
                        bottom: 12,
                        child: darkChip(
                          'SOURCE ${controller.cameraSource.toUpperCase()}  ·  ${controller.cameraFps} FPS',
                        ),
                      ),
                      Positioned(
                        right: 14,
                        bottom: 12,
                        child: Row(
                          children: [
                            darkChip(controller.formattedTime()),
                            const SizedBox(width: 7),
                            glassButton(
                              Icons.photo_camera_outlined,
                              liveCamera ? '实时画面' : '等待画面',
                              () => controller.onMessage(
                                liveCamera ? '当前显示 D435i 实时画面' : '当前没有可用的真实画面',
                              ),
                            ),
                            const SizedBox(width: 6),
                            glassButton(
                              controller.cameraExpanded
                                  ? Icons.fullscreen_exit
                                  : Icons.fullscreen,
                              '展开画面',
                              controller.toggleCameraExpanded,
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
            _buildTelemetry(),
          ],
        ),
      ),
    );
  }

  /// Camera-only metadata. Connection / latency / task count already live in
  /// the top status HUD, so they are not repeated here.
  Widget _buildTelemetry() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 6),
      child: Row(
        children: [
          _metric('SOURCE', controller.cameraLabel, mono: false),
          _metric('STATUS', controller.cameraStatus.toUpperCase()),
          _metric(
            'RESOLUTION',
            '${controller.cameraWidth}×${controller.cameraHeight}',
          ),
          _metric(
            'RUN STATE',
            controller.busy ? 'EXECUTING' : 'IDLE',
            valueColor: controller.busy
                ? ConsoleColors.accent
                : ConsoleColors.muted,
            last: true,
          ),
        ],
      ),
    );
  }

  Widget _metric(
    String label,
    String value, {
    Color? valueColor,
    bool last = false,
    bool mono = true,
  }) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12),
        decoration: BoxDecoration(
          border: last
              ? null
              : const Border(right: BorderSide(color: ConsoleColors.lineSoft)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                color: ConsoleColors.dim,
                fontSize: 9,
                fontWeight: FontWeight.w600,
                letterSpacing: 1,
              ),
            ),
            const SizedBox(height: 5),
            FittedBox(
              fit: BoxFit.scaleDown,
              alignment: Alignment.centerLeft,
              child: Text(
                value,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  color: valueColor ?? ConsoleColors.ink,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  fontFamily: mono ? 'monospace' : null,
                  fontFamilyFallback: const ['PingFang SC'],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _LiveCameraFrame extends StatefulWidget {
  const _LiveCameraFrame({
    required this.api,
    required this.framePath,
    required this.targetFps,
  });

  final ConsoleApi api;
  final String framePath;
  final int targetFps;

  @override
  State<_LiveCameraFrame> createState() => _LiveCameraFrameState();
}

class _LiveCameraFrameState extends State<_LiveCameraFrame> {
  Uint8List? _frame;
  int _generation = 0;

  @override
  void initState() {
    super.initState();
    _startFramePump();
  }

  @override
  void didUpdateWidget(covariant _LiveCameraFrame oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.api != widget.api ||
        oldWidget.framePath != widget.framePath ||
        oldWidget.targetFps != widget.targetFps) {
      _startFramePump();
    }
  }

  void _startFramePump() {
    final generation = ++_generation;
    unawaited(_pumpFrames(generation));
  }

  Future<void> _pumpFrames(int generation) async {
    final fps = widget.targetFps.clamp(1, 30);
    final frameInterval = Duration(microseconds: (1000000 / fps).round());
    while (mounted && generation == _generation) {
      final started = DateTime.now();
      try {
        final frame = await widget.api
            .fetchCameraFrame(widget.framePath)
            .timeout(const Duration(seconds: 2));
        if (!mounted || generation != _generation) return;
        if (frame.isNotEmpty) setState(() => _frame = frame);
      } catch (_) {
        if (!mounted || generation != _generation) return;
        await Future<void>.delayed(const Duration(milliseconds: 200));
      }

      final elapsed = DateTime.now().difference(started);
      final remaining = frameInterval - elapsed;
      if (remaining > Duration.zero) await Future<void>.delayed(remaining);
    }
  }

  @override
  void dispose() {
    _generation += 1;
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final frame = _frame;
    if (frame == null) {
      return const Center(
        child: SizedBox.square(
          dimension: 24,
          child: CircularProgressIndicator(
            strokeWidth: 2,
            color: ConsoleColors.accent,
          ),
        ),
      );
    }
    return RepaintBoundary(
      child: Image.memory(
        frame,
        fit: BoxFit.cover,
        gaplessPlayback: true,
        errorBuilder: (context, error, stackTrace) => const Center(
          child: Icon(
            Icons.broken_image_outlined,
            size: 36,
            color: ConsoleColors.dim,
          ),
        ),
      ),
    );
  }
}
