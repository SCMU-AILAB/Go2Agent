# Go2Agent 持续视觉互动 — 设计

日期：2026-10-14 · 状态：已确认

## 目标

让四足机器人（首个实现为 Unitree Go2）在开放场景中持续观察、理解任务并自主行动：

- **开放场景理解**：不依赖预设手势表。
- **任务泛化**：用自然语言（语音或 App）随时下新任务，无需改代码。
- **动作组合**：移动、转向、原生表演、说话可串成连贯行为。
- **跨机器人 / 跨模型**：换硬件只换 `robot` 实现，换模型只改配置。
- **反应快、不乱动**：闭环控制在本地高频运行，VLM 只做低频、事件驱动的决策。

第一阶段包含：持续视觉互动核心、语音下任务、TTS 回应、Swift 原生控制台 App。
非目标：VLA 训练、论文级数据采集、身份级 ReID、建图导航、公网鉴权。

## 部署

| 机器 | 运行 |
| --- | --- |
| GPU 服务器（RTX 3080 20G，局域网） | VLM（Qwen3-VL-8B 量化，vLLM，OpenAI 兼容 API）、感知服务（检测+跟踪）、ASR、TTS |
| 狗端主机 | `go2agent` 运行时：brain / executor / behaviors / safety / robot / sensors / audio / api |
| Mac / iOS | `frontend/` Swift 原生 App，通过 REST + WebSocket 连接狗端 `api` |

语言：Agent 与 SDK 用 Python 3.12；Rust 仅在实测瓶颈时引入；前端 Swift。

## 架构：慢脑 + 快反射

```
GPU 服务器: VLM | 感知(检测+跟踪 ~20Hz) | ASR | TTS
                 ▲ 局域网 HTTP / WebSocket
狗端: brain → executor → behaviors(20Hz) → safety → robot
      sensors(D435i RGB+深度) · audio · api ↔ Swift App
```

| 组件 | 职责 | 依赖 |
| --- | --- | --- |
| `core` | 共享数据类型（Frame、Track、Velocity、Event、RobotState、ActionSpec） | 无 |
| `robot` | 机器人接口：`velocity` / `perform` / `stop` / `state` / `actions`；Go2 实现 + Sim 实现 | core |
| `sensors` | 相机帧（RGB + 可选深度 + 时间戳）；RealSense、视频文件、Sim 实现 | core |
| `perception` | 跟踪结果来源：远程感知服务客户端；Sim 实现 | core |
| `safety` | 所有运动指令的唯一出口：限速、深度避停、看门狗、画面过期停 | robot, core |
| `behaviors` | 闭环行为原语，按 tick 运行，结束时给出结果 | core |
| `executor` | 行为队列、`keep/replace/append`、打断、发出行为事件 | behaviors |
| `brain` | 事件驱动调用 VLM，构造输入、校验输出 | core |
| `audio` | 麦克风→ASR→任务/急停关键词；文本→TTS→扬声器 | core |
| `api` | FastAPI：REST 下任务/急停，WebSocket 推送时间线与画面 | 运行时 |
| `timeline` | JSONL 运行日志 + 关键帧 JPEG + 单次思考回放 | core |
| `runtime` | 装配上述组件，20Hz 主循环 | 全部 |

## 慢脑（brain）

**触发**（事件驱动，不逐帧）：新任务 / 用户说话、行为成功或失败、目标丢失、有人出现或离开、障碍过近、
有活动行为时的定期复查（默认 4s）。同一时刻最多一个在途请求；在途期间到达的事件合并进下一次。

**输入**：任务文本；最近 1–3 个关键帧（约 448px），每个跟踪目标画编号框（set-of-marks）；
当前行为及进度；最近事件；机器人状态与可用动作清单；触发原因；上次输出若被拒绝则附原因。

**输出**（vLLM JSON schema 约束解码）：

```json
{
  "thought": "一句话推理",
  "say": "可选",
  "mode": "keep | replace | append",
  "plan": [{"behavior": "approach", "args": {"target": 3}}]
}
```

`keep` 为默认，维持当前行为不变（防止乱动的核心）；`replace` 打断并替换；`append` 接在后面。
目标通常用编号引用；检测器未覆盖时可给 `box`（归一化 xyxy），由感知服务据此建立新跟踪。

**校验**：编号不存在、动作不在清单、参数越界 → 拒绝，按 `keep` 处理，原因反馈到下次思考。

## 行为原语

| 行为 | 参数 | 结束条件 |
| --- | --- | --- |
| `follow` | `target`, `distance=1.2` | 取消，或目标丢失 |
| `approach` | `target`, `stop_distance=0.8` | 到达 |
| `face` | `target` | 目标居中 |
| `search` | `direction=left`, `timeout=8` | 出现新目标，或超时 |
| `move` | `dx`, `dy`, `dyaw` | 完成 |
| `perform` | `action`（来自 `robot.actions()`） | 动作完成 |
| `wait` | `seconds` | 到时 |
| `stop` | — | 立即 |

`say` 不入队，走独立通道，可边走边说。
跟随类行为目标短暂丢失时先由跟踪器尝试找回约 1s，失败后结束并报告 `target_lost`，由慢脑按描述重新指定目标。
距离优先用深度（目标框中心区域中位数），无深度时用框高度估计。

## 安全

- `safety` 是运动指令唯一出口：限速（默认 vx 0.5、vy 0.3、wz 0.8，可配）；前方深度 < 0.4m 时前进分量清零并发 `obstacle_close`；
  看门狗 200ms 无新指令即停；最新帧超过 0.5s 未更新即停。
- 动作清单带标签；`dangerous`（翻跟头、跳跃等）默认不进入慢脑可用清单，需配置显式开启。
- 急停走本地快速通道，不经过 VLM：App 停止按钮；语音关键词（停 / 别动 / stop）。
- 软件急停不替代遥控器；API 仅用于可信局域网。

## 故障处理

| 故障 | 行为 |
| --- | --- |
| VLM 超时（3s）/ 不可达 | 当前行为继续（本地闭环安全）；连续失败 > 10s 则完成当前行为后待机，并说明无法思考 |
| 感知服务断开 | 依赖跟踪的运动行为立即停止 |
| 慢脑输出不合理 | 拒绝，按 `keep`，原因反馈 |
| ASR / TTS 断开 | 仅影响语音，App 文字下任务照常 |

## 测试与调试

1. 单元测试：Sim 机器人 + Sim 传感器 + 固定 tick，覆盖行为、safety、executor 模式语义、brain 输出校验。
2. 慢脑离线评测：录制片段 + 任务 + 期望决策，批量跑 VLM，统计准确率与延迟；改提示词/换模型先跑评测。
3. 全链路仿真：Sim 机器人 + 视频文件。
4. 真机验收清单：跟随、走近打招呼、找人、急停、断网。

每次运行写 JSONL 时间线（事件、思考输入输出、行为切换、分段耗时）与关键帧 JPEG；
App 通过 WebSocket 实时显示；回放工具可把任意一次思考原样重新喂给 VLM。
