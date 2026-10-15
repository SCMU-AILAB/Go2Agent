import Foundation
import SwiftUI

// MARK: - Wire models mirroring docs/superpowers/specs/2026-10-14-app-protocol.md

struct Track: Codable, Identifiable, Equatable {
    let id: Int
    let label: String
    let box: [Double] // normalized x1, y1, x2, y2, origin top-left
    let score: Double
}

enum EventKind: String, Codable, CaseIterable {
    case task
    case userSpeech = "user_speech"
    case behaviorDone = "behavior_done"
    case behaviorFailed = "behavior_failed"
    case trackAppeared = "track_appeared"
    case trackLost = "track_lost"
    case obstacleClose = "obstacle_close"
    case periodic
    case estop
    case brainDecision = "brain_decision"
    case brainError = "brain_error"
    case say
    case service

    var title: String {
        switch self {
        case .task: "任务"
        case .userSpeech: "语音"
        case .behaviorDone: "行为完成"
        case .behaviorFailed: "行为失败"
        case .trackAppeared: "目标出现"
        case .trackLost: "目标丢失"
        case .obstacleClose: "障碍过近"
        case .periodic: "定期复查"
        case .estop: "急停"
        case .brainDecision: "思考决策"
        case .brainError: "思考失败"
        case .say: "说话"
        case .service: "服务状态"
        }
    }

    var icon: String {
        switch self {
        case .task: "square.and.pencil"
        case .userSpeech: "mic"
        case .behaviorDone: "checkmark.circle"
        case .behaviorFailed: "xmark.circle"
        case .trackAppeared: "person.crop.circle.badge.plus"
        case .trackLost: "person.crop.circle.badge.xmark"
        case .obstacleClose: "exclamationmark.triangle"
        case .periodic: "clock.arrow.circlepath"
        case .estop: "octagon"
        case .brainDecision: "brain"
        case .brainError: "brain.filled.badge.questionmark"
        case .say: "bubble.left"
        case .service: "dot.radiowaves.left.and.right"
        }
    }

    var isSerious: Bool {
        self == .behaviorFailed || self == .obstacleClose || self == .brainError
    }

    /// 事件语义色：时间线图标、卡片描边按此着色。
    var tint: Color {
        switch self {
        case .task: Theme.accent
        case .userSpeech: .purple
        case .behaviorDone: Theme.ok
        case .behaviorFailed: Theme.danger
        case .trackAppeared: .teal
        case .trackLost: .orange
        case .obstacleClose: Theme.caution
        case .periodic: Color.secondary
        case .estop: Theme.danger
        case .brainDecision: .indigo
        case .brainError: Theme.danger
        case .say: .mint
        case .service: Color.secondary
        }
    }
}

/// A generic JSON value so `data` payloads stay schema-flexible on the client.
enum JSONValue: Codable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case null
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            self = .null
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .number(let v): try container.encode(v)
        case .bool(let v): try container.encode(v)
        case .null: try container.encodeNil()
        case .array(let v): try container.encode(v)
        case .object(let v): try container.encode(v)
        }
    }

    var display: String {
        switch self {
        case .string(let v): v
        case .number(let v):
            v == v.rounded() && abs(v) < 1e12 ? String(Int(v)) : String(format: "%.2f", v)
        case .bool(let v): v ? "是" : "否"
        case .null: "null"
        case .array(let v): "[" + v.map(\.display).joined(separator: ", ") + "]"
        case .object(let v): "{" + v.map { "\($0.key): \($0.value.display)" }.joined(separator: ", ") + "}"
        }
    }

    var stringValue: String? {
        if case .string(let v) = self { return v }
        return nil
    }

    var numberValue: Double? {
        if case .number(let v) = self { return v }
        return nil
    }
}

struct RobotEvent: Codable, Equatable {
    let kind: EventKind
    let stamp: Double
    let data: [String: JSONValue]
}

struct RobotStatusPayload: Codable, Equatable {
    let connected: Bool
    let busy: Bool
}

struct ExecutorStatusPayload: Codable, Equatable {
    let current: String?
    let queue: [String]
    let runningS: Double?

    enum CodingKeys: String, CodingKey {
        case current
        case queue
        case runningS = "running_s"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        current = try c.decodeIfPresent(String.self, forKey: .current)
        queue = try c.decodeIfPresent([String].self, forKey: .queue) ?? []
        runningS = try c.decodeIfPresent(Double.self, forKey: .runningS)
    }

    init(current: String?, queue: [String], runningS: Double?) {
        self.current = current
        self.queue = queue
        self.runningS = runningS
    }
}

struct StatusSnapshot: Codable, Equatable {
    var estopped: Bool
    var task: String
    var robot: RobotStatusPayload
    var executor: ExecutorStatusPayload
    var tracks: [Track]
    var services: [String: Bool]
    var frameSeq: Int

    enum CodingKeys: String, CodingKey {
        case estopped
        case task
        case robot
        case executor
        case tracks
        case services
        case frameSeq = "frame_seq"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        estopped = try c.decodeIfPresent(Bool.self, forKey: .estopped) ?? false
        task = try c.decodeIfPresent(String.self, forKey: .task) ?? ""
        robot = try c.decodeIfPresent(RobotStatusPayload.self, forKey: .robot)
            ?? RobotStatusPayload(connected: false, busy: false)
        executor = try c.decodeIfPresent(ExecutorStatusPayload.self, forKey: .executor)
            ?? ExecutorStatusPayload(current: nil, queue: [], runningS: nil)
        tracks = try c.decodeIfPresent([Track].self, forKey: .tracks) ?? []
        services = try c.decodeIfPresent([String: Bool].self, forKey: .services) ?? [:]
        frameSeq = try c.decodeIfPresent(Int.self, forKey: .frameSeq) ?? 0
    }

    init(
        estopped: Bool = false, task: String = "",
        robot: RobotStatusPayload, executor: ExecutorStatusPayload,
        tracks: [Track] = [], services: [String: Bool] = [:], frameSeq: Int = 0
    ) {
        self.estopped = estopped
        self.task = task
        self.robot = robot
        self.executor = executor
        self.tracks = tracks
        self.services = services
        self.frameSeq = frameSeq
    }
}

struct PlanStep: Codable, Equatable {
    let behavior: String
    let args: [String: JSONValue]?

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        behavior = try c.decode(String.self, forKey: .behavior)
        args = try c.decodeIfPresent([String: JSONValue].self, forKey: .args)
    }

    var display: String {
        let rendered = (args ?? [:]).map { "\($0.key)=\($0.value.display)" }
            .sorted().joined(separator: ", ")
        return rendered.isEmpty ? behavior : "\(behavior)(\(rendered))"
    }
}

struct BrainDecision: Codable, Equatable {
    var thought: String
    var say: String
    var mode: String
    var plan: [PlanStep]
    var latencyMs: Double?
    var rejection: String?
    var frames: [String]?

    enum CodingKeys: String, CodingKey {
        case thought, say, mode, plan, rejection, frames
        case latencyMs = "latency_ms"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        thought = try c.decodeIfPresent(String.self, forKey: .thought) ?? ""
        say = try c.decodeIfPresent(String.self, forKey: .say) ?? ""
        mode = try c.decodeIfPresent(String.self, forKey: .mode) ?? "keep"
        plan = try c.decodeIfPresent([PlanStep].self, forKey: .plan) ?? []
        latencyMs = try c.decodeIfPresent(Double.self, forKey: .latencyMs)
        rejection = try c.decodeIfPresent(String.self, forKey: .rejection)
        frames = try c.decodeIfPresent([String].self, forKey: .frames)
    }

    init(
        thought: String = "", say: String = "", mode: String = "keep", plan: [PlanStep] = [],
        latencyMs: Double? = nil, rejection: String? = nil, frames: [String]? = nil
    ) {
        self.thought = thought
        self.say = say
        self.mode = mode
        self.plan = plan
        self.latencyMs = latencyMs
        self.rejection = rejection
        self.frames = frames
    }
}

// MARK: - Server messages

enum ServerMessage {
    case event(RobotEvent)
    case status(StatusSnapshot)

    static func decode(_ text: String) -> ServerMessage? {
        guard let data = text.data(using: .utf8) else { return nil }
        guard let envelope = try? JSONDecoder().decode(Envelope.self, from: data) else { return nil }
        switch envelope.type {
        case "event":
            guard let payload = try? JSONDecoder().decode(EventEnvelope.self, from: data) else { return nil }
            return .event(payload.event)
        case "status":
            guard let payload = try? JSONDecoder().decode(StatusEnvelope.self, from: data) else { return nil }
            return .status(payload.status)
        default:
            return nil
        }
    }
}

private struct Envelope: Decodable {
    let type: String
}

private struct EventEnvelope: Decodable {
    let event: RobotEvent
}

private struct StatusEnvelope: Decodable {
    let status: StatusSnapshot
}
