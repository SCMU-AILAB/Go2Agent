import CoreGraphics
import Foundation

/// Local fake feed so the console runs with zero backend: renders a synthetic scene,
/// moves a "person" around, and plays a scripted event timeline (10 fps).
@MainActor
final class MockFeed {
    private weak var store: ConsoleStore?
    private var runTask: Task<Void, Never>?

    private struct ScriptItem {
        let at: Double
        let kind: EventKind
        let data: [String: JSONValue]
    }

    private var script: [ScriptItem] = []
    private var scriptIndex = 0
    private var pendingTaskReply: [(Double, String)] = []

    func start(store: ConsoleStore) {
        self.store = store
        scriptIndex = 0
        script = Self.baseScript
        runTask = Task { [weak self] in
            await self?.run()
        }
    }

    func stop() {
        runTask?.cancel()
        runTask = nil
        store = nil
    }

    /// User submitted a task in mock mode: acknowledge with a decision after ~1.5s.
    func injectTask(_ text: String) {
        pendingTaskReply.append((currentTime + 1.5, text))
    }

    private var currentTime: Double = 0

    private func run() async {
        currentTime = 0
        while !Task.isCancelled {
            let t = currentTime
            if let store {
                let personX = 200.0 + 140.0 * sin(t * 0.6)
                let image = Self.scene(personX: personX)
                store.show(frame: FramePayload(seq: Int(t * 10), stamp: t, image: image))
                store.updateTracks([
                    Track(id: 1, label: "person", box: Self.personBox(cx: personX), score: 0.93),
                    Track(id: 2, label: "person", box: Self.personBox(cx: 500), score: 0.88),
                ])
                emitScript(t: t, store: store)
                emitTaskReplies(t: t, store: store)
            }
            currentTime += 0.1
            try? await Task.sleep(for: .milliseconds(100))
        }
    }

    private func emitScript(t: Double, store: ConsoleStore) {
        while scriptIndex < script.count, script[scriptIndex].at <= t {
            let item = script[scriptIndex]
            scriptIndex += 1
            store.append(event: RobotEvent(kind: item.kind, stamp: t, data: item.data))
        }
    }

    private func emitTaskReplies(t: Double, store: ConsoleStore) {
        guard !pendingTaskReply.isEmpty else { return }
        var remaining: [(Double, String)] = []
        for (at, text) in pendingTaskReply {
            if at <= t {
                store.append(event: RobotEvent(
                    kind: .brainDecision, stamp: t,
                    data: [
                        "thought": .string("收到任务「\(text)」，画面里 #1 在左侧活动，先转向再跟随"),
                        "say": .string("收到，\(text)"),
                        "mode": .string("replace"),
                        "plan": .array([
                            .object(["behavior": .string("face"), "args": .object(["target": .number(1)])]),
                            .object(["behavior": .string("follow"), "args": .object(["target": .number(1), "distance": .number(1.2)])]),
                        ]),
                        "latency_ms": .number(720),
                        "rejection": .null,
                    ]
                ))
            } else {
                remaining.append((at, text))
            }
        }
        pendingTaskReply = remaining
    }

    // MARK: - Scene rendering

    /// Person geometry in display space (16:9, 640x360, origin top-left).
    /// Track boxes are derived from the same constants so boxes always match the drawn figure.
    private static let personHalfWidth: Double = 26   // drawn body half-width
    private static let boxHalfWidth: Double = 30      // box slightly wider than the body
    private static let headTop: Double = 76           // display y of the head top
    private static let feet: Double = 230             // display y of the body bottom

    nonisolated private static func personBox(cx: Double) -> [Double] {
        [
            max(0, (cx - boxHalfWidth) / 640),
            headTop / 360,
            min(1, (cx + boxHalfWidth) / 640),
            feet / 360,
        ]
    }

    nonisolated private static func scene(personX: Double) -> CGImage {
        let w = CGFloat(640), h = CGFloat(360)
        let space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: Int(w), height: Int(h), bitsPerComponent: 8, bytesPerRow: 0,
            space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return blankImage()
        }
        // CGContext origin is bottom-left; the final image flips to top-left on display.
        // Wall region: context y in [horizon, h]; floor region: [0, horizon].
        let horizon = h / 3

        // 墙：上深下浅竖向渐变
        if let wall = CGGradient(
            colorsSpace: space,
            colors: [
                CGColor(srgbRed: 0.12, green: 0.14, blue: 0.19, alpha: 1),
                CGColor(srgbRed: 0.27, green: 0.30, blue: 0.36, alpha: 1),
            ] as CFArray,
            locations: [0.0, 1.0]
        ) {
            ctx.drawLinearGradient(wall, start: .zero, end: CGPoint(x: 0, y: h), options: [])
        }
        // 地板：上浅下深竖向渐变（裁剪到地平线以下）
        if let floor = CGGradient(
            colorsSpace: space,
            colors: [
                CGColor(srgbRed: 0.47, green: 0.45, blue: 0.42, alpha: 1),
                CGColor(srgbRed: 0.20, green: 0.19, blue: 0.18, alpha: 1),
            ] as CFArray,
            locations: [0.0, 1.0]
        ) {
            ctx.saveGState()
            ctx.clip(to: CGRect(x: 0, y: 0, width: w, height: horizon))
            ctx.drawLinearGradient(floor, start: CGPoint(x: 0, y: horizon), end: .zero, options: [])
            ctx.restoreGState()
        }
        // 踢脚线
        ctx.setFillColor(red: 0.08, green: 0.08, blue: 0.10, alpha: 1)
        ctx.fill(CGRect(x: 0, y: horizon - 6, width: w, height: 6))
        // 门：深色门扇 + 浅色门框 + 把手
        let door = CGRect(x: 90, y: horizon, width: 70, height: 150)
        ctx.setFillColor(red: 0.10, green: 0.11, blue: 0.14, alpha: 1)
        ctx.fill(door)
        ctx.setStrokeColor(red: 0.36, green: 0.39, blue: 0.45, alpha: 1)
        ctx.setLineWidth(2.5)
        ctx.stroke(door, width: 2.5)
        ctx.setFillColor(red: 0.78, green: 0.75, blue: 0.65, alpha: 1)
        ctx.fill(CGRect(x: 144, y: horizon + 66, width: 6, height: 6))
        // 人形
        drawPerson(ctx: ctx, cx: 500, hue: (0.10, 0.45, 0.30)) // 静态 #2
        drawPerson(ctx: ctx, cx: personX, hue: (0.75, 0.30, 0.20)) // 移动 #1
        // 暗角：四角压暗，聚焦中心
        if let vignette = CGGradient(
            colorsSpace: space,
            colors: [
                CGColor(srgbRed: 0, green: 0, blue: 0, alpha: 0),
                CGColor(srgbRed: 0, green: 0, blue: 0, alpha: 0.28),
            ] as CFArray,
            locations: [0.0, 1.0]
        ) {
            let center = CGPoint(x: w / 2, y: h / 2)
            ctx.drawRadialGradient(
                vignette,
                startCenter: center, startRadius: w * 0.30,
                endCenter: center, endRadius: w * 0.72,
                options: .drawsAfterEndLocation
            )
        }
        guard let image = ctx.makeImage() else { return blankImage() }
        return image
    }

    /// 人形：头（圆）+ 躯干（圆角矩形）+ 双腿。几何常量与 personBox 共用（显示空间）。
    nonisolated private static func drawPerson(ctx: CGContext, cx: Double, hue: (CGFloat, CGFloat, CGFloat)) {
        // CGContext y-axis points up; convert display-space geometry to context space.
        let cx = CGFloat(cx)
        let headDiameter = CGFloat(36)
        let torsoTop: CGFloat = 120  // display y of the torso top, just below the head
        let legTop: CGFloat = 190    // display y where legs begin

        let dark = (hue.0 * 0.72, hue.1 * 0.72, hue.2 * 0.72)

        // 双腿（暗一档）
        let legHeight = CGFloat(feet) - legTop
        ctx.setFillColor(red: dark.0, green: dark.1, blue: dark.2, alpha: 1)
        ctx.fill(CGRect(x: cx - CGFloat(personHalfWidth), y: 360 - CGFloat(feet), width: CGFloat(personHalfWidth) * 0.78, height: legHeight))
        ctx.fill(CGRect(x: cx + CGFloat(personHalfWidth) * 0.22, y: 360 - CGFloat(feet), width: CGFloat(personHalfWidth) * 0.78, height: legHeight))

        // 躯干：圆角矩形 + 深色勾边
        let torso = CGRect(
            x: cx - CGFloat(personHalfWidth),
            y: 360 - legTop,
            width: CGFloat(personHalfWidth) * 2,
            height: legTop - CGFloat(torsoTop)
        )
        let torsoPath = CGPath(
            roundedRect: torso,
            cornerWidth: 10,
            cornerHeight: 10,
            transform: nil
        )
        ctx.setFillColor(red: hue.0, green: hue.1, blue: hue.2, alpha: 1)
        ctx.addPath(torsoPath)
        ctx.fillPath()
        ctx.setStrokeColor(red: 0, green: 0, blue: 0, alpha: 0.22)
        ctx.setLineWidth(1.5)
        ctx.addPath(torsoPath)
        ctx.strokePath()

        // 头
        ctx.setFillColor(red: hue.0, green: hue.1, blue: hue.2, alpha: 1)
        ctx.fillEllipse(
            in: CGRect(
                x: cx - headDiameter / 2,
                y: 360 - (CGFloat(headTop) + headDiameter),
                width: headDiameter,
                height: headDiameter
            )
        )
    }

    nonisolated private static func blankImage() -> CGImage {
        let space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 0,
            space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        return ctx!.makeImage()!
    }

    private static let baseScript: [ScriptItem] = [
        ScriptItem(at: 0.5, kind: .service, data: ["service": .string("perception"), "ok": .bool(true), "detail": .string("20 Hz")]),
        ScriptItem(at: 1.0, kind: .task, data: ["text": .string("跟着穿深色衣服的人")]),
        ScriptItem(
            at: 2.5, kind: .brainDecision,
            data: [
                "thought": .string("任务要求跟随，#1 是左侧活动的人，转向并保持 1.2m 跟随"),
                "say": .string("好的，我跟上他"),
                "mode": .string("replace"),
                "plan": .array([
                    .object(["behavior": .string("face"), "args": .object(["target": .number(1)])]),
                    .object(["behavior": .string("follow"), "args": .object(["target": .number(1), "distance": .number(1.2)])]),
                ]),
                "latency_ms": .number(812),
                "rejection": .null,
            ]
        ),
        ScriptItem(at: 3.2, kind: .trackAppeared, data: ["id": .number(1), "label": .string("person")]),
        ScriptItem(at: 4.0, kind: .behaviorDone, data: ["behavior": .string("face"), "args": .object(["target": .number(1)]), "queue_empty": .bool(false)]),
        ScriptItem(at: 5.0, kind: .say, data: ["text": .string("你好，我过来啦")]),
        ScriptItem(at: 9.0, kind: .obstacleClose, data: ["distance": .number(0.35)]),
        ScriptItem(
            at: 12.0, kind: .brainDecision,
            data: [
                "thought": .string("前方 0.35m 有障碍，安全门已减速前进，继续跟随等待"),
                "say": .string(""),
                "mode": .string("keep"),
                "plan": .array([]),
                "latency_ms": .number(640),
                "rejection": .null,
            ]
        ),
        ScriptItem(at: 16.0, kind: .behaviorFailed, data: ["behavior": .string("follow"), "args": .object(["target": .number(1)]), "reason": .string("target_lost")]),
        ScriptItem(
            at: 18.0, kind: .brainDecision,
            data: [
                "thought": .string("目标短暂丢失，先原地搜索找回"),
                "say": .string(""),
                "mode": .string("replace"),
                "plan": .array([
                    .object(["behavior": .string("search"), "args": .object(["direction": .string("left"), "label": .string("person")])]),
                ]),
                "latency_ms": .number(700),
                "rejection": .null,
            ]
        ),
    ]
}
