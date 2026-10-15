import SwiftUI

/// 主控制台布局：顶栏（连接态 + 急停）· 相机 + 状态条 · 时间线 · 任务栏。
struct ConsoleView: View {
    let store: ConsoleStore
    @State private var showSettings = false
    @State private var taskDraft = ""

    var body: some View {
        VStack(spacing: 0) {
            topBar
            contentArea
            taskBar
        }
        #if os(macOS)
            .frame(minWidth: 960, minHeight: 620)
            .navigationTitle("Go2 控制台")
        #endif
        .sheet(isPresented: $showSettings) {
            SettingsView(store: store)
        }
        .onAppear {
            if store.connection == .disconnected {
                store.connect(mode: store.mode)
            }
        }
    }

    // MARK: - Top bar

    private var topBar: some View {
        HStack(spacing: 12) {
            // 连接态：呼吸圆点 + 数据源/地址 + 状态
            HStack(spacing: 9) {
                ConnectionDot(state: store.connection)
                VStack(alignment: .leading, spacing: 1) {
                    Text(store.mode == .mock ? "演示数据" : store.host)
                        .font(.callout.weight(.semibold))
                    Text(connectionText)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Theme.chipFill, in: Capsule())
            .overlay(Capsule().strokeBorder(Theme.cardStroke, lineWidth: 1))

            if let error = store.lastError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(Theme.danger)
                    .lineLimit(1)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(Theme.danger.opacity(0.12), in: Capsule())
            }

            Spacer(minLength: 8)

            Button {
                showSettings = true
            } label: {
                Label("设置", systemImage: "gearshape")
            }
            .buttonStyle(.bordered)
            .tint(.secondary)

            #if os(macOS)
            // iOS 上急停悬浮在相机右下角（见 cameraPanel）。
            EstopButton(estopped: store.estopped) {
                Task { await store.toggleEstop() }
            }
            #endif
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.bar)
        .overlay(alignment: .bottom) {
            Rectangle().fill(Theme.cardStroke).frame(height: 1)
        }
    }

    private var connectionText: String {
        switch store.connection {
        case .connected: "已连接"
        case .connecting: "连接中…"
        case .disconnected: "未连接"
        }
    }

    // MARK: - Content

    private var contentArea: some View {
        #if os(macOS)
        adaptiveContent
        #else
        GeometryReader { proxy in
            if proxy.size.width > 760 {
                adaptiveContent
            } else {
                compactContent
            }
        }
        #endif
    }

    /// 宽布局：左侧相机，右侧时间线。
    private var adaptiveContent: some View {
        HStack(spacing: 0) {
            cameraPanel
            Rectangle().fill(Theme.cardStroke).frame(width: 1)
            timelinePanel
                .frame(width: 340)
        }
    }

    /// 窄布局（iPhone）：相机在上，时间线填满剩余。
    private var compactContent: some View {
        VStack(spacing: 0) {
            cameraPanel
                .frame(height: 268)
            Rectangle().fill(Theme.cardStroke).frame(height: 1)
            timelinePanel
        }
    }

    private var cameraPanel: some View {
        VStack(spacing: 0) {
            CameraView(
                frame: store.frame,
                tracks: store.tracks,
                streamHealthy: store.streamHealthy
            )
            .aspectRatio(16 / 9, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
                    .strokeBorder(.primary.opacity(0.18), lineWidth: 1)
            )
            #if os(iOS)
            .overlay(alignment: .bottomTrailing) {
                EstopButton(estopped: store.estopped) {
                    Task { await store.toggleEstop() }
                }
                .controlSize(.small)
                .padding(10)
            }
            #endif
            .padding(.horizontal, 12)
            .padding(.top, 12)

            statusStrip
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
        }
    }

    private var statusStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                StatusChip(
                    label: "当前行为",
                    value: currentBehaviorText,
                    icon: "figure.walk",
                    ok: store.status?.executor.current != nil || store.mode == .mock
                )
                StatusChip(
                    label: "机器人",
                    value: robotText,
                    icon: "cpu",
                    ok: store.status?.robot.connected ?? true
                )
                StatusChip(
                    label: "感知",
                    value: servicesText("perception"),
                    icon: "eye",
                    ok: store.status?.services["perception"] ?? true
                )
                StatusChip(
                    label: "任务",
                    value: store.status?.task.isEmpty == false ? store.status!.task : "（无）",
                    icon: "text.bubble",
                    ok: store.status?.task.isEmpty == false
                )
            }
        }
    }

    private var currentBehaviorText: String {
        guard let executor = store.status?.executor else {
            return store.mode == .mock ? "follow #1（演示）" : "—"
        }
        if let current = executor.current {
            let seconds = executor.runningS.map { String(format: "，已运行 %.1fs", $0) } ?? ""
            return current + seconds
        }
        return "空闲"
    }

    private var robotText: String {
        guard let robot = store.status?.robot else {
            return store.mode == .mock ? "就绪（演示）" : "—"
        }
        if !robot.connected { return "未连接" }
        return robot.busy ? "执行原生动作中" : "就绪"
    }

    private func servicesText(_ service: String) -> String {
        guard let ok = store.status?.services[service] else {
            return store.mode == .mock ? "20 Hz（演示）" : "—"
        }
        return ok ? "正常" : "异常"
    }

    // MARK: - Timeline

    private var timelinePanel: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "clock.arrow.circlepath")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(Theme.accent)
                Text("时间线")
                    .font(.callout.weight(.semibold))
                Text("\(store.timeline.count)")
                    .font(.caption2.weight(.semibold).monospacedDigit())
                    .padding(.horizontal, 7)
                    .padding(.vertical, 2)
                    .background(Theme.chipFill, in: Capsule())
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            Rectangle().fill(Theme.cardStroke).frame(height: 1)
            TimelineView(entries: store.timeline)
        }
    }

    // MARK: - Task bar

    private var taskBar: some View {
        HStack(spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "square.and.pencil")
                    .font(.callout)
                    .foregroundStyle(.tertiary)
                TextField("输入新任务，例如「跟着我走」…", text: $taskDraft)
                    .textFieldStyle(.plain)
                    .onSubmit(submitTask)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .background(Theme.chipFill, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(Theme.accent.opacity(0.25), lineWidth: 1)
            )

            Button(action: submitTask) {
                Label("下达", systemImage: "paperplane.fill")
            }
            .buttonStyle(.plain)
            .font(.callout.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 9)
            .background(
                taskDraft.trimmingCharacters(in: .whitespaces).isEmpty
                    ? AnyShapeStyle(Color.secondary.opacity(0.25))
                    : AnyShapeStyle(Theme.accentGradient),
                in: RoundedRectangle(cornerRadius: 10, style: .continuous)
            )
            .disabled(taskDraft.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.bar)
        .overlay(alignment: .top) {
            Rectangle().fill(Theme.cardStroke).frame(height: 1)
        }
    }

    private func submitTask() {
        let text = taskDraft
        taskDraft = ""
        Task { await store.submitTask(text) }
    }
}

// MARK: - Connection dot

/// 连接状态圆点：外圈同色微光，内圈实心。
private struct ConnectionDot: View {
    let state: ConsoleStore.ConnectionState

    private var color: Color {
        switch state {
        case .connected: Theme.ok
        case .connecting: Theme.caution
        case .disconnected: Color.secondary
        }
    }

    var body: some View {
        ZStack {
            Circle()
                .fill(color.opacity(0.22))
                .frame(width: 20, height: 20)
            Circle()
                .fill(color)
                .frame(width: 9, height: 9)
        }
    }
}

// MARK: - Estop

/// 急停：渐变胶囊 + 呼吸光晕。点击立即生效，无确认弹窗（安全要求）。
struct EstopButton: View {
    let estopped: Bool
    let action: () -> Void
    @State private var pulsing = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: estopped ? "lock.open.fill" : "octagon.fill")
                    .font(.callout.weight(.semibold))
                Text(estopped ? "已急停 · 点击解除" : "急停")
                    .font(.callout.weight(.semibold))
            }
            .foregroundStyle(.white)
            .frame(minWidth: 104)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(
                estopped ? AnyShapeStyle(Theme.cautionGradient) : AnyShapeStyle(Theme.dangerGradient),
                in: Capsule()
            )
            .overlay(Capsule().strokeBorder(.white.opacity(0.25), lineWidth: 1))
            .shadow(
                color: (estopped ? Theme.caution : Theme.danger).opacity(pulsing ? 0.55 : 0.2),
                radius: pulsing ? 9 : 3,
                y: 1
            )
        }
        .buttonStyle(.plain)
        .keyboardShortcut(.cancelAction)
        .onAppear {
            withAnimation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true)) {
                pulsing = true
            }
        }
    }
}

// MARK: - Status chip

struct StatusChip: View {
    let label: String
    let value: String
    var icon: String?
    var ok = true

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 4) {
                if let icon {
                    Image(systemName: icon)
                        .font(.system(size: 9, weight: .semibold))
                }
                Text(label)
                    .font(.system(size: 10, weight: .medium))
            }
            .foregroundStyle(.tertiary)
            Text(value)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(ok ? Color.primary : Theme.danger)
                .lineLimit(1)
        }
        .frame(minWidth: 96, alignment: .leading)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(
            ok ? Theme.chipFill : Theme.danger.opacity(0.10),
            in: RoundedRectangle(cornerRadius: Theme.Radius.chip, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.chip, style: .continuous)
                .strokeBorder(ok ? Theme.cardStroke : Theme.danger.opacity(0.30), lineWidth: 1)
        )
    }
}

// MARK: - Settings

struct SettingsView: View {
    let store: ConsoleStore
    @Environment(\.dismiss) private var dismiss
    @State private var draftHost = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            header
            sourceSection
            hostSection
            Spacer(minLength: 0)
            buttons
        }
        .padding(20)
        #if os(macOS)
            .frame(width: 460)
        #endif
        .onAppear { draftHost = store.host }
    }

    private var header: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .fill(Theme.accentGradient)
                    .frame(width: 36, height: 36)
                Image(systemName: "gearshape.fill")
                    .foregroundStyle(.white)
                    .font(.headline)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("连接设置")
                    .font(.title3.weight(.semibold))
                Text("协议草案 2026-10-14")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private var sourceSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionLabel("数据源", icon: "square.stack.3d.up")
            Picker("数据源", selection: modeBinding) {
                ForEach(ConsoleStore.FeedMode.allCases, id: \.self) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .frame(width: 240)
            Text(store.mode == .mock ? "演示模式：本地合成数据，无需连接狗端" : "连接狗端的 REST 与 WebSocket 服务")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
    }

    private var hostSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionLabel("狗端地址", icon: "network")
            HStack(spacing: 6) {
                Text("http://")
                    .foregroundStyle(.tertiary)
                TextField("192.168.1.10:8200", text: $draftHost)
                    .textFieldStyle(.roundedBorder)
            }
            .disabled(store.mode == .mock)
            Text("需与机器狗处于同一局域网（端口 8200）")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
    }

    private var buttons: some View {
        HStack {
            Spacer(minLength: 0)
            Button("关闭") { dismiss() }
                .buttonStyle(.bordered)
                .keyboardShortcut(.cancelAction)
            Button("重新连接") {
                store.host = draftHost
                store.connect(mode: store.mode)
                dismiss()
            }
            .buttonStyle(.borderedProminent)
            .tint(Theme.accent)
        }
    }

    private func sectionLabel(_ text: String, icon: String) -> some View {
        Label(text, systemImage: icon)
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)
    }

    private var modeBinding: Binding<ConsoleStore.FeedMode> {
        Binding(
            get: { store.mode },
            set: { store.connect(mode: $0) }
        )
    }
}
