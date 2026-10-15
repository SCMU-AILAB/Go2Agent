# Go2Agent App 协议

日期：2026-10-15 · 状态：**以狗端实现为准**（`src/go2agent/api/server.py`）。前端按本文实现；狗端接口变化时同步更新本文。

狗端 `api` 默认 `http://<dog-host>:8080`（`go2agent run --api-port` 可改），无鉴权，仅限可信局域网。
狗当前在 `192.168.0.171`。

## 1. REST

| 方法 | 路径 | 请求体 | 响应 |
| --- | --- | --- | --- |
| GET | `/status` | — | `Status`（见 3.1），也用作连通性检查 |
| POST | `/task` | `{"text": "跟着我"}` | `Status`。空字符串 = 清除任务并停止；同时解除急停 |
| POST | `/hear` | `{"text": "..."}` | `{"ok": true}`。等同麦克风听到一句话（App 端语音识别可用） |
| POST | `/estop` | 无 | `Status`（`estopped: true`）。本地立即生效，不经 VLM，幂等 |
| POST | `/resume` | 无 | `Status`（`estopped: false`）。只解除急停，不改任务 |
| GET | `/frame.jpg` | — | 当前画面 JPEG（已画编号框）；无画面时 404 |

## 2. WebSocket

两条独立连接：

- **`/events`**（文本）：连接后第一条是 `{"kind": "status", "data": Status}`，之后推送每条时间线记录：
  `{"wall": 1792040000.12, "mono": 12345.67, "kind": "<EventKind>", "data": {...}}`。
  `wall` 是 Unix 时间，`mono` 是狗端单调时钟。不会定期推送状态：需要最新状态时轮询 `GET /status`（建议 1–2 s）。
- **`/video?fps=10`**（二进制）：每条消息是一张完整 JPEG（最长边 640，**已画好跟踪编号框**），没有额外帧头。
  `fps` 取 1–30，默认 10。只在有新画面时发送。

断线由 App 自行重连（建议指数退避 0.5 s → 5 s）。

## 3. 数据结构

### 3.1 Status

```json
{
  "task": "跟着穿黑衣服的人",
  "estopped": false,
  "thinking": false,
  "executor": {"current": "follow(target=3, distance=1.2)", "queue": ["perform(action='hello')"]},
  "robot": {"connected": true, "busy": false, "mode": 0, "gait_type": 0, "position": [0.1, 0.2, 0.3],
            "velocity": [0, 0, 0], "rpy": [0, 0, 0.6], "error_code": 100, "telemetry_age": 0.003,
            "last_error": null, "command_refusal": null, "stopping": false},
  "tracker_healthy": true,
  "tracks": [{"id": 3, "label": "person", "box": [0.40, 0.20, 0.60, 0.90]}],
  "last_decision": { ...与 brain_decision 的 data 相同... },
  "run_dir": "runs/20261015-220143"
}
```

- `executor.current` / `queue` 是行为的可读字符串（`名称(参数=值, ...)`）。
- `robot` 中除 `connected`、`busy` 外的字段来自具体机器人（Go2 遥测），可能缺省，App 不应依赖。
- `box` 为归一化 `x1, y1, x2, y2`（原点左上）。`/video` 的画面已经画了框，App 无需叠加；如要自绘，注意框与帧不保证同一时刻。

### 3.2 事件 kind 与 data（`core/types.py` 的 `EventKind`）

| kind | data |
| --- | --- |
| `task` | `{"text", "source"}`（`source`: `app` / `startup`） |
| `user_speech`, `say` | `{"text"}` |
| `behavior_done` | `{"behavior", "args", "reason", "queue_empty"}`（`reason` 可能是 `timeout` 等补充说明） |
| `behavior_failed` | `{"behavior", "args", "reason", "queue_empty"}`（`reason` 机器可读，如 `target_lost`、`action_failed: ...`） |
| `track_appeared`, `track_lost` | `{"id", "label"}` |
| `obstacle_close` | `{"direction", "distance"}`（`direction`: `front` / `back` / `left` / `right`，米，从车身边缘算） |
| `estop` | `{"source"}`（`app` / `voice`） |
| `service` | `{"service", "ok"}`（目前只有 `perception`；状态变化持续 1 s 才上报） |
| `brain_error` | `{"reason"}`，流式中断但已执行部分步骤时另有 `"partial": true` |
| `periodic` | `{}` |
| `brain_decision` | 见下 |

`brain_decision`：

```json
{
  "mode": "replace",
  "new_task": "",
  "plan": ["approach(target=3, stop_distance=0.8)", "perform(action='hello')"],
  "say": "你好",
  "thought": "目标3在挥手，走过去打招呼",
  "rejection": null,
  "ttft_ms": 450,
  "first_step_ms": 820,
  "latency_ms": 1390,
  "think_id": "0012"
}
```

- `plan` 是**已开始执行**的步骤的可读字符串（流式决策中被拒绝的后续步骤不在其中）。
- `rejection` 非 null 表示部分输出被校验拒绝（App 用警告色显示）。
- `first_step_ms` 是「请求 → 第一步开始执行」，即机器人感受到的延迟；`latency_ms` 是完整决策耗时。
- 思考时的输入关键帧不随事件推送；需要时在狗端 `runs/<run>/thinks/<think_id>-N.jpg` 查看。

## 4. 前端需要的改动（相对 frontend1st）

1. 端口 `8200` → `8080`。
2. `WS /stream` → 两条连接：`/events`（JSON，记录格式见 §2）和 `/video`（纯 JPEG，无帧头，已画框）。
3. `POST /release` → `POST /resume`；`/estop` 不需要请求体。
4. 没有 `/health`：用 `GET /status` 判断连通。
5. `/task`、`/estop`、`/resume` 返回完整 `Status`，可直接刷新界面。
6. 事件格式是 `{"wall","mono","kind","data"}`，不是 `{"type":"event","event":{...}}`；状态只在连接时推一次，之后轮询。
7. `brain_decision.plan` 是字符串数组，没有 `frames` 缩略图；新增 `new_task`、`ttft_ms`、`first_step_ms`、`think_id`。
8. 安全与稳定性（代码审查发现）：
   - 急停与解除不要用同一个按钮切换：解除急停应是单独操作并带确认（长按或二次确认），避免连点把急停立刻撤销。
   - `EventStream` 每次重连都新建 `URLSession` 且从不 `invalidateAndCancel()`，断续网络下会持续泄漏。
   - `EventStream` 的 `socket` 被多个任务读写，存在数据竞争；`MockFeed` 有 14 个 Swift 6 并发警告。
9. `.gitignore`：不要提交 `.dart_tool/`、Flutter 生成目录、`.workbuddy/` 等工具状态（frontend1st 曾把含 Cookies/Login Data 的
   Chrome 目录推到公开仓库，已从 `new` 历史中清除）。
