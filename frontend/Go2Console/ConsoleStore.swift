import CoreGraphics
import Foundation
import Observation

struct TimelineEntry: Identifiable {
    let id = UUID()
    let receivedAt = Date()
    let kind: EventKind
    let data: [String: JSONValue]

    var decision: BrainDecision? {
        guard kind == .brainDecision else { return nil }
        let object = JSONValue.object(data)
        guard let encoded = try? JSONEncoder().encode(object) else { return nil }
        return try? JSONDecoder().decode(BrainDecision.self, from: encoded)
    }

    var summary: String {
        switch kind {
        case .task, .userSpeech, .say:
            return data["text"]?.stringValue ?? ""
        case .behaviorDone:
            let reason = data["reason"]?.stringValue
            let suffix = reason.map { "（\($0)）" } ?? ""
            return behaviorLine + suffix
        case .behaviorFailed:
            return behaviorLine + "：" + (data["reason"]?.stringValue ?? "未知原因")
        case .trackAppeared, .trackLost:
            return "#\(data["id"]?.numberValue.map(Int.init) ?? 0) \(data["label"]?.stringValue ?? "")"
        case .obstacleClose:
            if let d = data["distance"]?.numberValue {
                return String(format: "前方 %.2f m，前进已暂停", d)
            }
            return "前方过近"
        case .estop:
            return "来源：\(data["source"]?.stringValue ?? "未知")"
        case .brainDecision:
            let mode = data["mode"]?.stringValue ?? "keep"
            let steps = (decision?.plan ?? []).map(\.display).joined(separator: " → ")
            let head = "mode=\(mode)"
            return steps.isEmpty ? head : "\(head)：\(steps)"
        case .brainError:
            return data["reason"]?.stringValue ?? ""
        case .periodic:
            return "定期复查"
        case .service:
            let name = data["service"]?.stringValue ?? ""
            let ok = (data["ok"]?.display ?? "?") == "是"
            return "\(name) \(ok ? "正常" : "异常")"
        }
    }

    private var behaviorLine: String {
        let behavior = data["behavior"]?.stringValue ?? "?"
        if let args = data["args"], case .object(let dict) = args, !dict.isEmpty {
            return behavior + "(" + dict.map { "\($0.key)=\($0.value.display)" }.sorted().joined(separator: ", ") + ")"
        }
        return behavior
    }
}

/// Central observable state. The only place that talks to APIClient / EventStream / MockFeed.
@MainActor
@Observable
final class ConsoleStore {
    enum ConnectionState {
        case disconnected
        case connecting
        case connected
    }

    enum FeedMode: String, CaseIterable {
        case mock
        case live

        var title: String {
            self == .mock ? "演示数据" : "连接狗端"
        }
    }

    private(set) var mode: FeedMode = .mock
    private(set) var connection: ConnectionState = .disconnected
    private(set) var streamHealthy = false
    private(set) var lastError: String?

    var host: String {
        didSet {
            UserDefaults.standard.set(host, forKey: "go2.host")
        }
    }

    private(set) var frame: CGImage?
    private(set) var tracks: [Track] = []
    private(set) var status: StatusSnapshot?
    private(set) var timeline: [TimelineEntry] = []
    private(set) var estopped = false
    private(set) var busy = false

    private var api: APIClient?
    private var stream: EventStream?
    private var mock: MockFeed?

    private static let timelineLimit = 400

    init() {
        host = UserDefaults.standard.string(forKey: "go2.host") ?? "192.168.1.10:8200"
    }

    // MARK: - Connection

    func connect(mode: FeedMode) {
        disconnect()
        self.mode = mode
        lastError = nil
        switch mode {
        case .mock:
            connection = .connected
            streamHealthy = true
            let mock = MockFeed()
            self.mock = mock
            mock.start(store: self)
        case .live:
            connection = .connecting
            let api = APIClient(host: host)
            self.api = api
            guard let url = URL(string: "ws://\(host.trimmingCharacters(in: .whitespaces))/stream") else {
                connection = .disconnected
                lastError = "无效的狗端地址"
                return
            }
            let stream = EventStream(store: self)
            self.stream = stream
            stream.start(url: url)
        }
    }

    func disconnect() {
        mock?.stop()
        mock = nil
        stream?.stop()
        stream = nil
        api = nil
        connection = .disconnected
        streamHealthy = false
    }

    // MARK: - Actions

    func submitTask(_ text: String) async {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        switch mode {
        case .mock:
            append(event: RobotEvent(kind: .task, stamp: 0, data: ["text": .string(text)]))
            mock?.injectTask(text)
        case .live:
            guard let api else { return }
            do {
                try await api.submitTask(text)
            } catch {
                lastError = "任务下达失败：\(error.localizedDescription)"
            }
        }
    }

    func toggleEstop() async {
        switch mode {
        case .mock:
            estopped.toggle()
            append(event: RobotEvent(kind: .estop, stamp: 0, data: ["source": .string("app")]))
        case .live:
            guard let api else { return }
            do {
                if estopped {
                    try await api.release()
                    estopped = false
                } else {
                    try await api.estop()
                    estopped = true
                }
            } catch {
                lastError = "急停操作失败：\(error.localizedDescription)"
            }
        }
    }

    // MARK: - Inbound (EventStream / MockFeed callbacks; all MainActor)

    func setStreamConnected(_ ok: Bool) {
        if ok {
            connection = .connected
            streamHealthy = true
            lastError = nil
        } else {
            streamHealthy = false
            if mode == .live { connection = .connecting }
        }
    }

    /// WS connection attempt failed; the stream keeps retrying with backoff.
    func streamConnectFailed() {
        if mode == .live {
            lastError = "无法连接狗端，正在自动重试…"
        }
    }

    func handle(_ message: ServerMessage) {
        switch message {
        case .event(let event):
            append(event: event)
        case .status(let snapshot):
            status = snapshot
            estopped = snapshot.estopped
            tracks = snapshot.tracks
        }
    }

    func show(frame payload: FramePayload) {
        frame = payload.image
    }

    func updateTracks(_ newTracks: [Track]) {
        tracks = newTracks
    }

    func append(event: RobotEvent) {
        timeline.append(TimelineEntry(kind: event.kind, data: event.data))
        if timeline.count > Self.timelineLimit {
            timeline.removeFirst(timeline.count - Self.timelineLimit)
        }
        if event.kind == .estop, let source = event.data["source"]?.stringValue, source != "app" {
            estopped = true
        }
    }
}
