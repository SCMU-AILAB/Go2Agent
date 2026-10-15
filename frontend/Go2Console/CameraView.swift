import SwiftUI

/// 实时相机画面 + set-of-marks 编号跟踪框叠加。
struct CameraView: View {
    let frame: CGImage?
    let tracks: [Track]
    let streamHealthy: Bool

    /// 跟踪框调色板：按 track id 取模分配，颜色之间拉开色相。
    private static let palette: [Color] = [
        Color(red: 0.31, green: 0.62, blue: 1.00),
        Color(red: 0.27, green: 0.80, blue: 0.55),
        Color(red: 1.00, green: 0.66, blue: 0.24),
        Color(red: 0.70, green: 0.52, blue: 1.00),
        Color(red: 1.00, green: 0.48, blue: 0.68),
        Color(red: 0.24, green: 0.80, blue: 0.80),
    ]

    var body: some View {
        ZStack {
            // 无画面时的深色渐变底
            LinearGradient(
                colors: [
                    Color(red: 0.07, green: 0.08, blue: 0.10),
                    Color(red: 0.11, green: 0.12, blue: 0.15),
                ],
                startPoint: .top,
                endPoint: .bottom
            )

            if let frame {
                Image(frame, scale: 1, label: Text("相机画面"))
                    .resizable()
                    .scaledToFit()
            } else {
                emptyState
            }
        }
        // 跟踪框叠加：Canvas 绘制，坐标按归一化 box 映射
        .overlay {
            GeometryReader { proxy in
                let size = proxy.size
                Canvas { context, _ in
                    guard size.width > 0, frame != nil else { return }
                    for track in tracks {
                        guard track.box.count == 4 else { continue }
                        let rect = CGRect(
                            x: track.box[0] * size.width,
                            y: track.box[1] * size.height,
                            width: (track.box[2] - track.box[0]) * size.width,
                            height: (track.box[3] - track.box[1]) * size.height
                        )
                        let color = Self.palette[abs(track.id) % Self.palette.count]
                        context.stroke(
                            Path(roundedRect: rect, cornerRadius: 4),
                            with: .color(color),
                            style: StrokeStyle(lineWidth: 2.5)
                        )
                        // 编号标签：圆角色块贴在框顶外侧
                        let label = "#\(track.id)"
                        let labelRect = CGRect(
                            x: rect.minX, y: max(0, rect.minY - 17),
                            width: 30, height: 17
                        )
                        context.fill(
                            Path(roundedRect: labelRect, cornerRadius: 4),
                            with: .color(color)
                        )
                        context.draw(
                            Text(label)
                                .font(.system(size: 10, weight: .bold))
                                .foregroundColor(.white),
                            in: labelRect
                        )
                    }
                }
            }
            .allowsHitTesting(false)
        }
        // 顶部渐隐：保证叠加在其上的信息条可读
        .overlay(alignment: .top) {
            LinearGradient(
                colors: [.black.opacity(0.35), .clear],
                startPoint: .top,
                endPoint: .bottom
            )
            .frame(height: 36)
            .allowsHitTesting(false)
        }
        .overlay(alignment: .bottomLeading) {
            if !tracks.isEmpty {
                Text(tracks.map { "#\($0.id) \($0.label)" }.joined(separator: " · "))
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(.ultraThinMaterial, in: Capsule())
                    .padding(10)
            }
        }
        .overlay(alignment: .bottomTrailing) {
            if !streamHealthy {
                Label("画面中断", systemImage: "wifi.slash")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background(Theme.danger, in: Capsule())
                    .shadow(color: Theme.danger.opacity(0.5), radius: 5, y: 2)
                    .padding(10)
            }
        }
    }

    /// 无画面时的空态。
    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: streamHealthy ? "video" : "video.slash")
                .font(.system(size: 30, weight: .light))
                .foregroundStyle(.white.opacity(0.5))
            Text(streamHealthy ? "等待画面…" : "画面中断")
                .font(.callout.weight(.medium))
                .foregroundStyle(.white.opacity(0.7))
            Text(streamHealthy ? "WebSocket 帧流连接正常" : "检查狗端连接或网络")
                .font(.caption)
                .foregroundStyle(.white.opacity(0.4))
        }
    }
}
