import 'package:flutter/material.dart';

import '../../../../core/theme/console_colors.dart';
import '../../controllers/console_controller.dart';
import 'console_widgets.dart';

class VoicePanel extends StatelessWidget {
  const VoicePanel({super.key, required this.controller});

  final ConsoleController controller;

  @override
  Widget build(BuildContext context) {
    final active = controller.voiceEnabled;
    final statusColor = controller.voiceStatus == 'error'
        ? ConsoleColors.red
        : active
        ? ConsoleColors.green
        : ConsoleColors.dim;
    return panel(
      header: sectionTitle(Icons.record_voice_over_outlined, 'VOICE'),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          dot(statusColor, glow: active),
          const SizedBox(width: 6),
          Text(
            controller.voiceStatus.toUpperCase(),
            style: TextStyle(
              color: statusColor,
              fontSize: 10,
              fontWeight: FontWeight.w700,
              letterSpacing: .8,
            ),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: double.infinity,
              height: 40,
              child: OutlinedButton.icon(
                onPressed: controller.backend ? controller.toggleVoice : null,
                icon: Icon(
                  active ? Icons.mic_off_outlined : Icons.mic_none_outlined,
                  size: 16,
                ),
                label: Text(active ? '停止语音对话' : '启动语音对话'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: active
                      ? ConsoleColors.muted
                      : ConsoleColors.accent,
                  backgroundColor: ConsoleColors.field,
                  side: BorderSide(
                    color: active
                        ? ConsoleColors.line
                        : ConsoleColors.accent.withValues(alpha: .4),
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(6),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            telemetryRow(
              'MIC',
              controller.voiceListening
                  ? 'LISTENING'
                  : controller.voiceInputDevice,
              valueColor: controller.voiceListening
                  ? ConsoleColors.green
                  : ConsoleColors.muted,
            ),
            telemetryRow(
              'SPEAKER',
              controller.voiceTtsEngine ?? controller.voiceOutputDevice,
            ),
            const SizedBox(height: 8),
            _speechBlock('YOU', controller.voiceTranscript, '等待语音输入…'),
            const SizedBox(height: 8),
            _speechBlock('GO2', controller.voiceReply, '等待 Agent 回复…'),
            if (controller.voiceError != null) ...[
              const SizedBox(height: 8),
              Text(
                controller.voiceError!,
                style: const TextStyle(color: ConsoleColors.red, fontSize: 11),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _speechBlock(String label, String text, String placeholder) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: ConsoleColors.field,
        border: Border.all(color: ConsoleColors.lineSoft),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(
              color: ConsoleColors.dim,
              fontSize: 9,
              fontWeight: FontWeight.w700,
              letterSpacing: 1,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            text.isEmpty ? placeholder : text,
            maxLines: 3,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              color: text.isEmpty ? ConsoleColors.dim : ConsoleColors.ink,
              fontSize: 11,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }
}
