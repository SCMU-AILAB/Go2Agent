import Foundation
import ImageIO

struct FramePayload {
    let seq: Int
    let stamp: Double
    let image: CGImage
}

/// WebSocket client for `/stream` (protocol §2): text = status + events, binary = JPEG frames.
/// Reconnects with exponential backoff (0.5s -> 5s) until `stop()`.
final class EventStream {
    private let store: ConsoleStore
    private var url: URL?
    private var socket: URLSessionWebSocketTask?
    private var runTask: Task<Void, Never>?
    private var shouldRun = false

    init(store: ConsoleStore) {
        self.store = store
    }

    func start(url: URL) {
        stop()
        self.url = url
        shouldRun = true
        runTask = Task { [weak self] in
            await self?.loop()
        }
    }

    func stop() {
        shouldRun = false
        runTask?.cancel()
        runTask = nil
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
    }

    private func loop() async {
        var backoff: UInt64 = 500_000_000
        while shouldRun {
            if let url, let ws = await connectOnce(url: url) {
                await store.setStreamConnected(true)
                await receiveLoop(ws)
                await store.setStreamConnected(false)
                backoff = 500_000_000
            } else {
                await store.streamConnectFailed()
            }
            guard shouldRun else { break }
            try? await Task.sleep(nanoseconds: backoff)
            backoff = min(backoff * 2, 5_000_000_000)
        }
    }

    private func connectOnce(url: URL) async -> URLSessionWebSocketTask? {
        let session = URLSession(configuration: .ephemeral)
        let ws = session.webSocketTask(with: url)
        socket = ws
        ws.resume()
        let ok: Bool = await withCheckedContinuation { continuation in
            ws.sendPing { error in
                continuation.resume(returning: error == nil)
            }
        }
        if ok {
            return ws
        }
        ws.cancel()
        socket = nil
        return nil
    }

    private func receiveLoop(_ ws: URLSessionWebSocketTask) async {
        while shouldRun {
            do {
                let message = try await ws.receive()
                switch message {
                case .string(let text):
                    if let decoded = ServerMessage.decode(text) {
                        await store.handle(decoded)
                    }
                case .data(let data):
                    if let frame = Self.decodeFrame(data) {
                        await store.show(frame: frame)
                    }
                @unknown default:
                    break
                }
            } catch {
                return
            }
        }
    }

    /// Binary frame format: uint32 BE seq | float64 BE stamp | JPEG bytes.
    static func decodeFrame(_ data: Data) -> FramePayload? {
        let headerLength = 4 + 8
        guard data.count > headerLength else { return nil }
        let seq = data.prefix(4).reduce(0) { ($0 << 8) | Int($1) }
        let stampBits = data.subdata(in: 4..<12).reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
        let stamp = Double(bitPattern: stampBits)
        let jpeg = data.dropFirst(headerLength)
        guard let source = CGImageSourceCreateWithData(jpeg as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        return FramePayload(seq: seq, stamp: stamp, image: image)
    }
}
