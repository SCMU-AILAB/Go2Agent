import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';
import 'console_widgets.dart';

/// Operator-adjustable gesture confirmation window (seconds).
class VisionHoldPanel extends StatefulWidget {
  const VisionHoldPanel({super.key, required this.controller});

  final ConsoleController controller;

  @override
  State<VisionHoldPanel> createState() => _VisionHoldPanelState();
}

class _VisionHoldPanelState extends State<VisionHoldPanel> {
  late final TextEditingController _input;
  bool _dirty = false;

  @override
  void initState() {
    super.initState();
    _input = TextEditingController(
      text: widget.controller.visionConfirmHoldS.toStringAsFixed(2),
    );
  }

  @override
  void didUpdateWidget(covariant VisionHoldPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!_dirty && _input.text.isEmpty) {
      _input.text = widget.controller.visionConfirmHoldS.toStringAsFixed(2);
    }
  }

  @override
  void dispose() {
    _input.dispose();
    super.dispose();
  }

  Future<void> _apply() async {
    final raw = _input.text.trim();
    final value = double.tryParse(raw);
    if (value == null || value < 0 || value > 30) {
      widget.controller.onMessage('请填写 0～30 之间的秒数');
      return;
    }
    await widget.controller.applyVisionConfirmHold(value);
    if (!mounted) return;
    setState(() {
      _dirty = false;
      _input.text = value.toStringAsFixed(2);
    });
  }

  @override
  Widget build(BuildContext context) {
    final c = widget.controller;
    return panel(
      header: sectionTitle(Icons.timer_outlined, '手势确认时长'),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              '持续识别到同一手势达到该秒数后才执行；可减少误触发。',
              style: TextStyle(color: ConsoleColors.dim, fontSize: 11, height: 1.4),
            ),
            const SizedBox(height: 10),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _input,
                    keyboardType:
                        const TextInputType.numberWithOptions(decimal: true),
                    onChanged: (_) => setState(() => _dirty = true),
                    style: const TextStyle(color: ConsoleColors.ink, fontSize: 14),
                    cursorColor: ConsoleColors.accent,
                    decoration: InputDecoration(
                      hintText: '例如 1.5',
                      suffixText: '秒',
                      helperText:
                          '当前生效：${c.visionConfirmHoldS.toStringAsFixed(2)} 秒',
                      fillColor: ConsoleColors.field,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                SizedBox(
                  height: 40,
                  child: FilledButton(
                    onPressed: _dirty ? _apply : null,
                    style: FilledButton.styleFrom(
                      backgroundColor: ConsoleColors.accent,
                      foregroundColor: ConsoleColors.bg0,
                      disabledBackgroundColor: ConsoleColors.bg2,
                      disabledForegroundColor: ConsoleColors.faint,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(6),
                      ),
                    ),
                    child: const Text('应用'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
