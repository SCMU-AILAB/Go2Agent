import SwiftUI

/// 事件时间线。`brain_decision` 行就地展开思考详情。
/// 自动跟随最新事件；向上翻看历史时暂停跟随，右下角出现「最新」按钮。
struct TimelineView: View {
    let entries: [TimelineEntry]
    @State private var followLatest = true

    var body: some View {
        Group {
            if entries.isEmpty {
                ContentUnavailableView(
                    "暂无事件",
                    systemImage: "list.bullet.rectangle",
                    description: Text("连接后事件会实时出现在这里")
                )
            } else {
                ScrollViewReader { proxy in
                    let list = List {
                        ForEach(entries) { entry in
                            TimelineRow(entry: entry)
                                .id(entry.id)
                                .listRowInsets(EdgeInsets(top: 4, leading: 12, bottom: 4, trailing: 12))
                                .listRowSeparator(.hidden)
                                .listRowBackground(Color.clear)
                        }
                    }
                    .listStyle(.plain)
                    .scrollContentBackground(.hidden)
                    .onChange(of: entries.last?.id) { _, last in
                        guard followLatest, let last else { return }
                        withAnimation(.easeOut(duration: 0.15)) {
                            proxy.scrollTo(last, anchor: .bottom)
                        }
                    }
                    if #available(macOS 15.0, iOS 18.0, *) {
                        list
                            .onScrollGeometryChange(for: Bool.self) { geometry in
                                let maxOffset = geometry.contentSize.height
                                    - geometry.containerSize.height
                                    - geometry.contentInsets.bottom
                                return geometry.contentOffset.y >= maxOffset - 60
                            } action: { _, isNearBottom in
                                followLatest = isNearBottom
                            }
                            .overlay(alignment: .bottomTrailing) {
                                if !followLatest {
                                    jumpToLatest(proxy: proxy)
                                }
                            }
                    } else {
                        // 老系统：始终跟随最新事件（原行为）。
                        list
                    }
                }
            }
        }
    }

    private func jumpToLatest(proxy: ScrollViewProxy) -> some View {
        Button {
            followLatest = true
            if let last = entries.last?.id {
                withAnimation(.easeOut(duration: 0.2)) {
                    proxy.scrollTo(last, anchor: .bottom)
                }
            }
        } label: {
            Image(systemName: "arrow.down")
                .font(.caption.weight(.bold))
                .foregroundStyle(.white)
                .frame(width: 30, height: 30)
                .background(Theme.accentGradient, in: Circle())
                .shadow(color: Theme.accent.opacity(0.4), radius: 6, y: 2)
        }
        .buttonStyle(.plain)
        .padding(12)
    }
}

// MARK: - Row

private struct TimelineRow: View {
    let entry: TimelineEntry
    @State private var expanded = false

    private var timeText: String {
        entry.receivedAt.formatted(date: .omitted, time: .standard)
    }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            // 事件类型图标章
            ZStack {
                Circle()
                    .fill(entry.kind.tint.opacity(0.14))
                    .frame(width: 30, height: 30)
                Image(systemName: entry.kind.icon)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(entry.kind.tint)
            }

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(entry.kind.title)
                        .font(.subheadline.weight(.semibold))
                    Spacer(minLength: 4)
                    Text(timeText)
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.tertiary)
                    if entry.kind == .brainDecision {
                        Image(systemName: expanded ? "chevron.up" : "chevron.down")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.tertiary)
                    }
                }
                Text(entry.summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(expanded ? nil : 2)
                if expanded, let decision = entry.decision {
                    ThinkDetail(decision: decision)
                }
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous)
                .fill(Theme.cardFill)
        )
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous)
                .strokeBorder(rowStroke, lineWidth: 1)
        )
        .contentShape(Rectangle())
        .onTapGesture {
            if entry.kind == .brainDecision { expanded.toggle() }
        }
    }

    /// 卡片描边按事件类型着色：决策=强调蓝，严重事件=语义红/橙，其余=发丝线。
    private var rowStroke: Color {
        if entry.kind == .brainDecision { return Theme.accent.opacity(0.35) }
        if entry.kind.isSerious { return entry.kind.tint.opacity(0.35) }
        return Theme.cardStroke
    }
}

// MARK: - Think detail

private struct ThinkDetail: View {
    let decision: BrainDecision

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // 输入关键帧缩略图
            if let frames = decision.frames, !frames.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Array(frames.enumerated()), id: \.offset) { _, dataURL in
                            ThumbnailView(dataURL: dataURL)
                                .frame(width: 96, height: 54)
                                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                                        .strokeBorder(.white.opacity(0.2), lineWidth: 1)
                                )
                        }
                    }
                }
            }

            // 计划步骤：编号芯片
            if !decision.plan.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("计划")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.tertiary)
                    ForEach(Array(decision.plan.enumerated()), id: \.offset) { index, step in
                        HStack(spacing: 7) {
                            Text("\(index + 1)")
                                .font(.system(size: 9, weight: .bold, design: .rounded))
                                .foregroundStyle(.white)
                                .frame(width: 16, height: 16)
                                .background(Theme.accent, in: Circle())
                            Text(step.display)
                                .font(.caption.weight(.medium))
                                .foregroundStyle(Theme.accent)
                                .lineLimit(1)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Theme.accent.opacity(0.10), in: Capsule())
                    }
                }
            }

            LabeledRow(label: "推理", value: decision.thought)
            if !decision.say.isEmpty {
                LabeledRow(label: "说话", value: decision.say)
            }
            LabeledRow(
                label: "耗时",
                value: decision.latencyMs.map { String(format: "%.0f ms", $0) } ?? "—"
            )
            if let rejection = decision.rejection {
                LabeledRow(label: "被拒绝", value: rejection, serious: true)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Theme.chipFill)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Theme.accent.opacity(0.15), lineWidth: 1)
        )
    }
}

private struct LabeledRow: View {
    let label: String
    let value: String
    var serious = false

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Text(label)
                .font(.caption.weight(.medium))
                .foregroundStyle(.tertiary)
                .frame(width: 38, alignment: .leading)
            Text(value)
                .font(.caption)
                .foregroundStyle(serious ? Theme.danger : .primary)
            Spacer(minLength: 0)
        }
    }
}

/// Decodes a `data:image/...;base64,...` URL into a thumbnail image.
private struct ThumbnailView: View {
    let dataURL: String
    @State private var image: CGImage?

    var body: some View {
        Group {
            if let image {
                Image(image, scale: 1, label: Text("思考输入帧"))
                    .resizable()
                    .scaledToFill()
            } else {
                Rectangle().fill(.quaternary)
            }
        }
        .task(id: dataURL) {
            image = Self.decode(dataURL: dataURL)
        }
    }

    nonisolated private static func decode(dataURL: String) -> CGImage? {
        guard let comma = dataURL.range(of: ",") else { return nil }
        let base64 = String(dataURL[comma.upperBound...])
        guard let data = Data(base64Encoded: base64),
              let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}
