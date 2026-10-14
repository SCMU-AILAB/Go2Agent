# Go2Agent

四足机器人（首个实现：Unitree Go2）的持续视觉互动 Agent。设计见
[docs/superpowers/specs/2026-10-14-vision-agent-design.md](docs/superpowers/specs/2026-10-14-vision-agent-design.md)。

**慢脑 + 快反射**：VLM 只在有新情况时思考，输出「行为」（跟随、走近、转向、找人、移动、原生动作）；
行为在狗端以 20Hz 闭环执行，所有运动经过唯一的安全闸。

```
GPU 服务器 (3080 20G): VLM (vLLM) | 感知 (YOLO+ByteTrack) | ASR | TTS
          ▲ 局域网
狗端: brain → executor → behaviors (20Hz) → safety → robot
      sensors (D435i) · audio · api ↔ App
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/go2agent/core` | 共享数据类型 |
| `src/go2agent/robot` | 机器人接口；`sim.py` 模拟，`go2.py` 真机，`go2_actions.py` 动作清单 |
| `src/go2agent/sensors` | 相机：RealSense、视频文件 / 设备 |
| `src/go2agent/perception` | 跟踪客户端与线协议 |
| `src/go2agent/behaviors` | 行为原语与几何 |
| `src/go2agent/executor.py` | 行为队列（keep / replace / append） |
| `src/go2agent/safety.py` | 安全闸：限速、深度避停、看门狗、急停 |
| `src/go2agent/brain` | VLM 客户端、提示词、输出 schema、校验、离线评测 |
| `src/go2agent/runtime.py` | 装配与主循环 |
| `src/go2agent/api` | 给 App 的 REST + WebSocket |
| `src/go2agent/server` | GPU 服务器上的感知 / ASR / TTS 服务 |
| `deploy/gpu-server` | VLM 启动脚本 |
| `frontend/` | 原生 App（另行开发） |

## 运行

```bash
uv sync                       # 开发机
uv run pytest                 # 全部测试，无需硬件 / 模型

# GPU 服务器
deploy/gpu-server/serve-vlm.sh                  # VLM，:8000，模型名 brain
uv sync --extra server && uv run go2agent-server all --port 8100   # 感知 + ASR + TTS

# 狗端（真机）
uv run go2agent run --server <GPU服务器IP> --robot go2 --camera realsense

# 不连真狗：模拟机器人 + 视频文件
uv run go2agent run --server <GPU服务器IP> --robot sim --camera video:clip.mp4 --task "跟着穿黑衣服的人"
```

真机测试时遥控器必须在手；软件急停（App 按钮、语音「停」）不能替代遥控器。API 无鉴权，只在可信局域网使用。

## 调试慢脑

每次运行写到 `runs/<时间>/`：`timeline.jsonl` 记录全部事件与决策（含耗时），`thinks/NNNN.json`
是每次思考发给 VLM 的原样输入（图片另存为 JPEG）。

```bash
uv run go2agent replay runs/<run>/thinks/0003.json --server <IP> -n 5 --current-prompt   # 重放 / 测延迟
uv run go2agent eval evals/cases.jsonl --server <IP> --current-prompt                   # 准确率 + 延迟
```

改提示词或换模型前后各跑一次 `eval`，以数据决定。用例格式见 `src/go2agent/brain/evaluate.py`。

## App API（狗端 :8080）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/status` | 任务、急停、当前行为与队列、跟踪目标、上次决策、机器人状态 |
| POST | `/task` `{"text"}` | 设置任务（空字符串清除并停止）；同时解除急停 |
| POST | `/hear` `{"text"}` | 注入一句话，等同麦克风听到（可用于 App 端语音识别） |
| POST | `/estop` | 急停（锁定） |
| POST | `/resume` | 解除急停，不改任务 |
| GET | `/frame.jpg` | 当前画面（带编号框） |
| WS | `/events` | 首条 `{"kind":"status","data":...}`，之后推送每条时间线记录 `{"wall","mono","kind","data"}` |
| WS | `/video?fps=10` | 二进制 JPEG 帧（带编号框） |

时间线记录的 `kind` 取值见 `core/types.py` 的 `EventKind`；`brain_decision` 的 `data` 含
`thought`、`say`、`task`、`mode`、`plan`、`rejection`、`latency_ms`、`think_id`。
