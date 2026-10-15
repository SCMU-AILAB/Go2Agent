# Go2Agent App 协议 — 前后端契约草案

日期：2026-10-14 · 状态：草案（前端按此实现，狗端 `api` 实现时以此为准，改动需同步更新本文档）

所有传输都走狗端 `api` 服务，默认 `http://<dog-host>:8200`，仅限可信局域网。

## 1. REST

| 方法 | 路径 | 请求体 | 响应 |
| --- | --- | --- | --- |
| GET | `/health` | — | `{"status": "ok"}` |
| GET | `/status` | — | `StatusSnapshot`（见 2.3） |
| POST | `/task` | `{"text": "跟着我"}` | `{"ok": true}` |
| POST | `/estop` | `{"source": "app"}` | `{"ok": true, "estopped": true}` |
| POST | `/release` | `{}` | `{"ok": true, "estopped": false}` |

- `/task`：下达新任务，立即返回，思考结果走事件流。
- `/estop`：急停，不经 VLM，本地立即生效（安全通道）。幂等。
- `/release`：解除急停。幂等。

## 2. WebSocket `/stream`

一个连接同时承载：状态快照、事件流（文本）与画面帧流（二进制）。

### 2.1 文本消息（服务端 → 客户端，JSON）

事件（与 `core/types.py` 的 `Event.to_json()` 对齐）：

```json
{"type": "event", "event": {"kind": "behavior_done", "stamp": 12345.67, "data": {"behavior": "face", "args": {...}, "queue_empty": true}}}
```

状态快照（连接时先推一次，之后事件驱动 + 每 2s 定期）：

```json
{"type": "status", "status": {"estopped": false, "task": "...", "robot": {"connected": true, "busy": false}, "executor": {"current": "follow #1", "queue": [], "running_s": 3.2}, "tracks": [...], "services": {"vlm": true, "perception": true, "asr": false, "tts": true}, "frame_seq": 1024}}
```

### 2.2 二进制消息（服务端 → 客户端，画面帧）

```
| uint32 BE seq | float64 BE stamp | JPEG bytes (RGB, 16:9) |
```

- `stamp` 为狗端 `time.monotonic()` 秒，与事件 `stamp` 同源（仅用于服务端侧排序，App 显示用本地接收时间）。
- 帧率约 10–15 fps；客户端始终显示最新帧，跳帧即丢弃。

### 2.3 数据结构

`Track`（对齐 `core/types.py`）：

```json
{"id": 1, "label": "person", "box": [0.10, 0.17, 0.21, 0.67], "score": 0.92}
```

`box` 为归一化 `x1, y1, x2, y2`（origin 左上）。跟踪框由 App 本地叠加绘制（帧图像本身不带框，便于复用）。

事件 `kind` 全集（对齐 `EventKind`）：`task` / `user_speech` / `behavior_done` / `behavior_failed` / `track_appeared` / `track_lost` / `obstacle_close` / `periodic` / `estop` / `brain_decision` / `brain_error` / `say` / `service`。

各 `kind` 的 `data`：

| kind | data |
| --- | --- |
| `task`, `user_speech`, `say` | `{"text": "..."}` |
| `behavior_done` | `{"behavior", "args", "reason", "queue_empty"}` |
| `behavior_failed` | `{"behavior", "args", "reason"}`（reason 为机器可读原因，如 `target_lost`） |
| `track_appeared`, `track_lost` | `{"id", "label"}` |
| `obstacle_close` | `{"distance"}`（米） |
| `estop` | `{"source"}`（`app` / `voice` / `api`） |
| `service` | `{"service", "ok", "detail"}` |
| `brain_error` | `{"reason"}` |
| `periodic` | `{}` |

`brain_decision`（思考详情，供 App 展开回放）：

```json
{
  "thought": "任务要求跟随，#1 是同一个人，继续跟",
  "say": "收到",
  "mode": "keep",
  "plan": [{"behavior": "follow", "args": {"target": 1, "distance": 1.2}}],
  "latency_ms": 812.5,
  "rejection": null,
  "frames": ["data:image/jpeg;base64,..."]
}
```

- `frames`：本次思考的输入关键帧缩略图（≤256px、quality 60、最多 3 张），仅在 `brain_decision` 事件携带，避免常态流量膨胀。
- `rejection`：非 null 表示该输出被校验拒绝（App 显示为警告色）。
- `mode`：`keep` / `replace` / `append`。

### 2.4 客户端 → 服务端

暂无（连接即全量订阅）。预留 `{"type": "subscribe", ...}` 扩展（如按需关帧流）。连接断开由客户端自动重连（指数退避 0.5s → 5s）。

## 3. App 端行为约定

- 急停按钮：点击立即 POST `/estop`，**无确认弹窗**；急停态下再点即 POST `/release`。
- 事件时间线：本地环形缓冲保留最近 400 条；`brain_decision` 可展开详情。
- `stamp` 是服务端单调时钟，App 显示相对时间一律用本地接收时刻。
- 本地网络 HTTP：App 需在 Info.plist 声明 `NSAllowsLocalNetworking`（iOS 另需 `NSLocalNetworkUsageDescription`）。
