# Go2 Agent

以 **Robot Skill Runtime** 为唯一动作执行边界。文本、麦克风和视觉都不能直接分发
机器人动作。仓库专门支持 Go2，文本与视觉 Agent 共享完整的 Go2 技能注册表。

Go2 侧已提供：`UnitreeGo2Adapter` / `UnitreeGo2Config`、Go2 专用装配路径
（CLI / FastAPI / perception）、`register_go2_skills()`、周期刷新的移动技能、
原生动作目录（比心/跳舞等）以及文本/持续视觉任务模式。Go2 语音使用主机外接麦克风和
扬声器，通过本地主机 TTS 播报。详见 [Go2 Adapter 说明](docs/go2-adapter.md)。

## 控制台任务模式（文本 / 持续视觉）

新版控制台任务面板显式选择任务模式：

| 模式 | 行为 |
| --- | --- |
| **文本指令**（默认） | 把指令交给文本 Agent 调工具；开启真实相机也可发送「比心」「坐下」「跳舞」 |
| **持续视觉交互** | 视觉 Agent 根据实时画面、目标和 SkillResult 选工具；必须 `cameraSource=local` |

- API 为兼容旧客户端仍使用 `taskMode="gesture"` 表示持续视觉模式。
- Go2 视觉模式不依赖固定手势标签；模型按任务从当前注册且允许的 Skill 目录选工具。
  不会凭提示词发明未注册 Skill，具体动作仍受本地安全前提限制。
- 文本 Agent **不接收图像**，即使预览开着也不会“看见”人或障碍。
- 相机预览正常 ≠ 已选择文本模式。

```json
{"instruction":"给我比个心","cameraSource":"local","taskMode":"text"}
{"instruction":"有人打招呼就打招呼，有人比心就比心","cameraSource":"local","taskMode":"gesture"}
```

## Go2 原生动作目录（`register_go2_skills`）

只注册与 Go2 SportClient 语义一致的 Skill。SDK `status=0` 仅表示
命令被接受，**不代表物理动作完成**；是否支持取决于真机固件。

**常用目录（文本 Agent / 视觉 Agent / API 可见）**

| 类别 | Skills |
| --- | --- |
| 姿态 | `stand_up` / `stand_down` / `sit` / `rise_sit` / `balance_stand` |
| 原生表演 | `hello` / `stretch` / `content` / `heart`（比心） / `scrape` / `dance1` / `dance2` |
| 视觉别名 | `wave` → 底层 `hello`（不是人形手臂挥手） |
| 开关（布尔 `flag`） | `pose`（`{"flag":true\|false}`） |
| 短距移动 | `move_forward` / `move_backward` / `move_left` / `move_right` / `turn_left` / `turn_right` / `move` |
| 停止 | `stop` / `stop_move` |

**控制与特技目录（同样默认注册）**

- 危险动作：`damp`、`front_flip`、`front_jump`、`front_pounce`、`left_flip`、`back_flip`，以及 flag 类 `hand_stand` / `free_bound` / `free_jump` / `walk_upright` / `switch_joystick`
- 步态/模式：`recovery_stand`、`free_walk`、`static_walk`、`trot_run`、`economic_gait`、`switch_avoid_mode`，以及 `free_avoid` / `classic_walk` / `cross_step` / `auto_recover_set`

`register_go2_skills()` 默认注册全部 46 个技能，文本 Agent、视觉 Agent 和 API
使用同一 Registry。`operator_only` 和 `dangerous` 标签是动作提示，不是视觉目录过滤器；
参数校验、资源锁、感知前提和深度停止继续生效。

文本 Agent 不依赖一份固定的中文动作映射；它读取当前 Registry 自动生成的工具描述和
参数 schema，再按用户目标选择一个或多个 Skill。`local_commands` 仅是无模型离线备用模式，
才使用有限的显式命令匹配。完整说明见 `src/agent/service.py` 的 `GO2_SYSTEM_PROMPT`
与 [Go2 Adapter](docs/go2-adapter.md)。

**遥测**：Go2 Adapter 订阅 `rt/sportmodestate`，`get_state().details` 含 `mode`、
`gait_type`、位置/速度/姿态、`error_code`；`telemetry_available` 仅在最近 2 秒有数据时为 true。

**语音**：Go2 使用主机 TTS；`VuiClient` 只有开关/音量/亮度，不能当 TTS。

## 一键启动 Go2 控制台

此脚本连真机但**不会**自动提交动作任务。默认语音把转写结果作为持续视觉目标，
5070 Ti 负责视觉决策，不需要另备文本 Ollama：

```bash
sh scripts/run-go2-console.sh
# 默认：真机 + 本地相机 + 外接麦克风/扬声器 + vision 语音目标
# 网卡可用 GO2_NETWORK 覆盖；视觉服务用 GO2_VISION_URL 覆盖
```

## 控制台视觉接入

Flutter 控制台已接入滑动视频策略。选择「持续视觉交互」并使用本地相机时，才会启动
持续视觉决策；选择「文本指令」时只把文字交给 Agent。Go2 的视觉 Agent 每轮读取
任务、最近视频窗口、当前动作和上一次 SkillResult，可选择执行技能、说话、继续、
打断或忽略；已注册且允许的技能由模型按任务选择，不限于预设手势。相机预览与模型共享旋转后的
JPEG，决策、Skill 结果与状态通过 REST/WebSocket 快照展示。详见
[前端运行说明](frontend/README.md)。

```bash
# 局域网 UnifoLM 只负责视觉；此命令使用模拟机器人，不会下发真机动作
.venv/bin/python -m app.api --camera-source local \
  --vision-backend unifolm \
  --vision-model unitreerobotics/UnifoLM-ER-1 \
  --vision-url http://192.168.31.112:8011 --no-audio
```

不要同时启动占用同一相机的 `run-remote-vision.sh`。前端选择视觉模式并开始任务才会
启动视觉决策；停止任务会结束 worker。连接真机由后端 `--hardware --network eth0` 决定，
软件停止不是物理急停，API 仅用于可信网络。

Go2 的 `front_jump` 等动作也会进入完整视觉目录。视觉模型按任务与画面选择技能，
模型输出仍需通过本地技能目录和参数校验；服务接受动作命令不等于动作已经完成。

Go2 目录包含 `follow_person`：前端选择「本地相机」和
「持续视觉交互」，输入“跟着前面的人走，保持距离”。视觉 Agent 决定是否
开始/继续/中断；狗端 D435i 根据单个人体框中心及对齐深度，以 0.20--0.30 m/s
前进、最多 0.3 rad/s 转向，目标约 1.5 m。目标丢失、多人、深度无效、画面
超过 0.5 秒未更新或障碍过近均停止。停止任务/急停也会取消动作。
它还不是身份追踪或避障导航；请在开阔平地、遥控器在手的条件下验收。
需要完全离线的有限动作词表时，可改用 `GO2_VOICE_AGENT=local_commands`。

Go2 真机直接使用启动脚本。视觉和默认语音目标都连接局域网 UnifoLM；
若改用 Ollama 文本 Agent，需配置可用的 `OLLAMA_HOST`：

```bash
GO2_VISION_URL=http://192.168.31.112:8011 sh scripts/run-go2-console.sh
```

文本/麦克风路径如下：

```text
文本输入 ──────────────────────┐
                              v
麦克风 -> Whisper ASR -> LangChain Agent -> 最终回复 -> 主机 TTS
                              |
                              v
                       LangChain tools
                              |
                              v
                         SkillRuntime
                              |
                         RobotSkill
                              |
                         UnitreeGo2Adapter
                              |
               unitree_sdk2_cpp bindings
```

当前视觉入口使用滑动视频窗口策略；原来的稀疏事件 Agent 仍可作为
回退模式：

```text
                              ┌-> VideoBuffer -> Video VLM -> AgentDecision
RealSense -> CameraFrame -----|
                              └-> HOG/depth safety
                                      |
                 AgentDecision -> SkillRuntime -> Go2
```

相机持续采集 30 FPS。CLI 默认保留最近 2 秒并取最新 1 帧，控制台默认保留
0.8 秒并取 3 帧；视频策略以 500 ms 为目标间隔。VLM 推理、Skill 执行和相机采集相互解耦，
模型不会逐帧运行。如果一次推理超过 500 ms，策略不会并发堆积请求，而是在本次
推理完成后再开始下一次。HOG/WorldEvent 保留用于可观测性，中心区域深度安全停止
不等待 VLM。

## 目录

```text
src/
├── agent/       # 对话 Agent、事件 Decision Agent 和决策执行闭环
├── app/         # 文本/麦克风 CLI 入口
├── adapters/    # LangChain tools、Whisper 输入、主机语音输出
├── core/        # Skill Runtime 核心协议与执行器
├── perception/  # D435i 取流、人员检测、最小状态和事件检测
├── robot/       # RobotAdapter、Go2 SDK 适配器、模拟适配器、factory
└── skills/      # Go2 Robot Skill（go2_catalog）

frontend/        # Flutter 控制台（暗色 Mission Control UI）
```

## 单技能验收

完全绕过 Agent 和模型，通过 Runtime 直接执行 Go2 技能：

```bash
uv run go2-action hello
uv run go2-action pose --arguments '{"flag":true}'
```

默认使用模拟 Go2；真机调用增加 `--hardware --network eth0`。
SkillExecutor 在同一资源锁和超时内运行 `execute() -> verify() -> cleanup()`；
Go2 原生姿态成功表示 SDK 接受命令，不保证物理动作已完成。

## 安装

安装项目依赖并准备 Ollama：

```bash
uv sync
ollama serve
ollama pull <model-name>
```

机器人主机已经有 Unitree Python bindings 时，直接从现有目录安装：

```bash
uv pip install -e ~/unitree_sdk2/unitree_sdk2_bindings
```

如果当前环境没有 `uv`，也可以使用机器人上运行项目的 Python：

```bash
python3 -m pip install -e ~/unitree_sdk2/unitree_sdk2_bindings
```

代码直接使用 bindings 中的：

- `unitree_sdk2_cpp.channel.initialize/release`
- `unitree_sdk2_cpp.robot.go2.SportClient`
- `unitree_sdk2_cpp.idl.go2.SportModeState`

## Flutter 控制台 FastAPI 后端

`go2-api` 提供与仓库内 `frontend/` 的 `ConsoleController` 状态字段对齐的 REST 和
WebSocket 接口。默认绑定 `0.0.0.0:8000`、使用模拟机器人，并在进程启动时自动
创建后端会话：

```bash
uv run go2-api
```

真机运行时增加 `--hardware`；TTS 使用主机外接扬声器：

```bash
uv run go2-api \
  --hardware \
  --network eth0
```

若只需要真机动作、不需要扬声器，可增加 `--no-audio`。D435i 通过 USB 连接运行
后端的主机，启动时选择真实相机：

```bash
uv run go2-api --camera-source local
```

主要接口：

```text
GET    /api/v1/health
GET    /api/v1/console
POST   /api/v1/session/start
POST   /api/v1/session/stop
PUT    /api/v1/config/system-prompt
POST   /api/v1/tasks
POST   /api/v1/tasks/current/cancel
GET    /api/v1/skills
POST   /api/v1/skills/{name}/execute
PUT    /api/v1/camera/source
GET    /api/v1/camera/frame.jpg
DELETE /api/v1/logs
POST   /api/v1/voice/start
POST   /api/v1/voice/stop
POST   /api/v1/voice/speak
WS     /api/v1/events
```

API JSON 使用 camelCase，核心状态可直接映射前端的 `backend`、`starting`、
`busy`、`promptSaved`、`sessionId`、`cameraSource`、`modelStatus`、
`skillStatus`、`skillName`、`progressText`、`modelOutput`、`modelDuration`、
`latency`、`voice`、`tools` 和 `logs`。`voice` 包含监听状态、最新转写、最新回复和
STT/TTS 错误。WebSocket 连接后首先返回完整 `state`，之后持续发送
`state`、`log`、`heartbeat` 和 `camera` 事件。

例如提交任务：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tasks \
  -H 'content-type: application/json' \
  -d '{"instruction":"跟我挥挥手","cameraSource":"demo"}'
```

所有 API 触发的动作仍只走：

```text
FastAPI -> RobotAgent / SkillRuntime -> RobotSkill -> RobotAdapter -> Go2
```

API 路由不会直接调用 Unitree SDK。

## 文本入口

无硬件时使用模拟 `RobotAdapter`，便于验证 Agent 和工具调用：

```bash
uv run go2agent --input text
```

文本会先送入 LangChain `create_agent`。例如用户要求挥手时，Agent 调用 `wave`
工具，工具只调用 `SkillRuntime.execute()`，不会生成或解析 action 字符串。
增加 `--no-audio` 可禁用主机 TTS。

## 狗端本地语音对话

Go2 使用连接到狗端电脑的外接麦克风和扬声器：Faster Whisper 模型常驻进程完成
STT，Piper 完成本地 TTS；未提供 Piper 模型时回退 `espeak-ng`。录音与播放共享
半双工锁，机器人发声时不会同时录音。

在狗端现有虚拟环境中安装，不要执行会覆盖硬件依赖的普通 `uv sync`：

```bash
sudo apt install alsa-utils espeak-ng
.venv/bin/python -m pip install faster-whisper piper-tts
```

第一次运行 Faster Whisper 会下载模型。可提前预热：

```bash
.venv/bin/python -c \
  "from faster_whisper import WhisperModel; WhisperModel('small', device='auto', compute_type='default')"
```

FastAPI 与前端一起使用：

```bash
.venv/bin/python -m app.api \
  --hardware --network eth0 \
  --camera-source local --vision-rotation-deg 0 \
  --vision-backend unifolm --vision-url http://192.168.31.112:8011 \
  --voice --voice-agent-backend vision --record-seconds 3 \
  --whisper-model /home/cf/Go2Agent/models/faster-whisper-small \
  --whisper-device cpu --whisper-compute-type int8 \
  --audio-input-device pulse --audio-output-device pulse \
  --piper-model /home/cf/Go2Agent/models/piper/zh_CN-huayan-medium.onnx \
  --host 0.0.0.0 --port 8000
```

如果尚未放置 Piper 中文 `.onnx` 和同名 `.onnx.json`，去掉 `--piper-model`，系统会
使用 `espeak-ng`。不加 `--voice` 时也可以在前端的 VOICE 面板手动启动监听。

`vision` 模式下，ASR 识别的自然语言是持续视觉 Agent 的目标，不经过固定动作
关键词表；每次新语音目标会先停止旧任务，明确的停止指令直接本地中断。
例如说“跟着前面的人走”会让视觉模型从实时画面和技能目录中选择
`follow_person`，狗端再做深度反馈控制。该模式需要 D435i 和可用的视觉服务。
语音路径为：

```text
外接麦克风 -> Faster Whisper -> 持续视觉 Agent -> SkillRuntime -> Go2
                                      ^             |
                                      |             v
                                  最新画面       SkillResult
                                      |             |
                                      +-------------+
                        简短回应 -> Piper -> 外接扬声器
```

备用 `local_commands` 支持明确的比心、打招呼、坐下、站起、趴下、伸懒腰、跳舞、短距离
移动/转向和停止指令；未知语句不会猜动作。回复是按实际 `SkillResult` 生成的确定性短句，
不是开放式聊天。需要独立文本 LLM 时可改为 `--voice-agent-backend shared`，但不要让它
和单 in-flight 的视觉 UnifoLM 共用同一服务。机器人动作始终只能通过 `SkillRuntime`。

### 旧 Whisper CLI 入口

麦克风输入通过 `arecord` 或 `ffmpeg` 录音，再交给本地 Whisper CLI：

```bash
sudo apt install alsa-utils ffmpeg

uv run go2agent --input microphone \
  --whisper-bin whisper-cli \
  --whisper-model /opt/models/ggml-base.bin \
  --language zh \
  --record-seconds 5
```

USB 麦克风不是默认设备时增加 `--audio-device hw:2,0`。ASR 的输出只是普通
用户文本，后续路径与文本入口完全一致。

## 连接真机

```bash
uv run go2agent \
  --hardware \
  --network eth0 \
  --input microphone \
  --whisper-bin whisper-cli \
  --whisper-model /opt/models/ggml-base.bin \
  --language zh
```

指定 `--hardware` 后程序会直接连接机器人。连接顺序为：

```text
初始化 DDS -> SportClient.init() -> 主机 TTS 初始化 -> Agent 循环
```

Agent 的最终文字回复通过主机 Piper 或 espeak-ng 播放。`--no-audio` 禁用语音输出。
`connect()` 不下发运动命令，`close()` 只释放通信资源；移动技能在清理时请求停止。

## D435i 视觉闭环

### 4090D 远程推理

#### Jetson Orin NX 8GB 本地 INT4（实验）

官方 BF16 checkpoint 实测常驻约 8.8 GiB，无法稳定放入 8GB 统一内存。`test`
分支提供 `--vision-backend llamacpp`，用于运行量化后的 UnifoLM GGUF；视觉输出仍经过
原有 Pydantic Schema、SkillRegistry、时效门和深度安全，不能直接调用 SDK。

先准备支持 Qwen3-VL 多模态的 `llama-server`、Q4_K_M 主模型和 mmproj，然后启动：

```bash
export UNIFOLM_MODEL_GGUF=/path/to/UnifoLM-ER-1-Q4_K_M.gguf
export UNIFOLM_MMPROJ_GGUF=/path/to/mmproj-UnifoLM-ER-1-Q8_0.gguf
sh scripts/run-jetson-unifolm-server.sh
```

另一终端先用模拟机器人验收 Schema 和时延：

```bash
sh scripts/run-jetson-unifolm-vision.sh --once
```

现场检查后再连接 Go2：

```bash
sh scripts/run-jetson-unifolm-vision.sh --hardware --network eth0
```

这是 8GB PoC，不承诺准确率或实时性。运行时用 `tegrastats` 检查统一内存；不要同时
常驻大型 Whisper、检测器和另一套 VLM。若出现 OOM，先缩短上下文、减少视觉帧数，
不能通过放宽决策时效来掩盖慢推理。

#### 宇树 UnifoLM-ER-1（Go2，可选）

仓库保留文字 Agent 的 Ollama/Qwen 链路，并用 `--vision-backend unifolm` 单独连接
视觉服务。当前局域网服务位于 `http://192.168.31.112:8011`。UnifoLM 对 JSON 提示
曾使用单标签手势协议；当前 Go2 默认改为开放的结构化动作决策。模型每轮读取
视觉目标、最新画面、D435i 本地观测、可用 Skill 目录和执行历史，再输出执行、
说话、继续、打断或忽略。模型输出不能跳过本地 Skill、深度和时效校验。

视觉服务器健康检查：

```bash
curl http://192.168.31.112:8011/health
```

Go2 端启动完整控制台：

```bash
sh scripts/run-go2-console.sh
```

另一终端先只测相机和模型，不驱动机器人：

```bash
sh scripts/run-unifolm-vision.sh --once
```

现场安全检查并准备急停后再连接 Go2：

```bash
sh scripts/run-unifolm-vision.sh --hardware --network eth0
```

2026-09-18 实测当前相机空场景返回 `none`，服务端推理约 0.11 秒，HTTP 往返约
0.16 秒，并被狗端判定为 `ignore`。这只验证传输、格式和安全拒绝链路，不代表真实
挥手/比耶/比心识别准确率；真机动作前仍需现场逐项验证。
Go2 当前可直接映射 `wave` 和 `heart`。`handshake`、`high_five` 若不在 Go2
SkillRegistry 中会被拒绝，不会绕过 Runtime 调用不存在的动作。

单标签 UnifoLM 路径不生成视觉回复文本；语音输入与文字 Agent 使用本地
Faster Whisper、独立 Ollama 和 Piper。
2026-09-12 带语音回归：补充六字段完整输出及“走近、手臂下垂不是握手”的说明后，
同一23窗口回放动作选择23/23、无格式错误、无无动作场景误触发；该集已用于调试，
不是独立准确率评测。结果保存于 `debug/vision/eval-speaking-20260912-v2-retry.jsonl`。
该次云端往返中位数6.826秒，部分响应中 `load_s` 占主要耗时，实时性能尚待排查；
不要提高动作时效上限来绕过过期拦截。
`execute_and_speak` 会并发启动 Skill 与主机 TTS，让动作和语音同时进行；
因此动作最终失败时，已经开始的语音不会回滚。相同动作即使文字不同也共享冷却限制，
未确认或正在进行的动作不重复说话。
增加 `--no-audio` 可静音但保留模型生成文字。日志 `speech_spoken=true` 表示 TTS 调用
成功返回，不代表已通过麦克风验证声音播放完成。

远端 `go2-vision-ollama.service` 是当前用户的临时 systemd 服务，监听
`127.0.0.1:11435`，远程脚本默认模型为 `qwen3.5:9b`，使用 `egocentric` 第一视角
提示词并关闭思考输出。保留原 `qwen2.5vl:3b` 供回退。不需要修改 frpc 或开放推理公网端口。
服务器重启后需要重新启动该服务：

```bash
systemd-run --user --unit=go2-vision-ollama \
  --setenv=OLLAMA_HOST=127.0.0.1:11435 \
  --setenv=OLLAMA_NUM_PARALLEL=1 /usr/local/bin/ollama serve
```

机器人端先在一个终端保持隧道运行（已有隧道时不要重复启动）：

```bash
sh scripts/remote-vision-tunnel.sh
```

另一终端先验证相机与远程模型，不驱动机器人：

```bash
sh scripts/run-remote-vision.sh --once
```

调试识别时保存实际发往模型的 JPEG 帧（不连接机器人）：

当前安装的相机画面倒置，远程脚本默认 `--vision-rotation-deg 180`，仅旋转 VLM RGB
输入，抓帧保存旋转后的输入。深度安全和原有人员检测不变。若重新安装相机使其正立，
请覆盖为 `--vision-rotation-deg 0`，避免再次倒置。

```bash
sh scripts/run-remote-vision.sh --vision-capture-dir debug/vision --vision-capture-limit 20
```

默认不抓帧；开启后每次运行创建独立会话目录，最多保存 20 个窗口，达到上限后
停止保存，推理继续，不自动删除旧文件。每个窗口包含按时间排序的 `frame-*.jpg`、
带时间戳及哈希的 `input.json`、对应决策或错误的 `result.json`。
决策日志的 `model_metrics.capture_path` 指向该窗口；`agent.vision_capture` 也会打印保存路径。
这是模型输入和决策记录，不代表动作执行成功。图像仅新增本地副本，不额外上传；
可能包含人脸等隐私信息，按需保留。默认目录已加入 Git 忽略；多次启动仍会累计占用磁盘。

完成现场安全检查并备好急停后使用真机：

```bash
sh scripts/run-remote-vision.sh --hardware --network eth0
```

图片和模型提示词通过 SSH 加密发送至服务器，本地保留深度安全和 SkillRuntime。
远程脚本默认启用 `--vision-task social`：最近 0.8 秒取 3 帧，使用开放
`AgentDecision` 结构化输出，从完整 Go2 Registry 选择技能。模糊、遮挡、非面向机器人、
最新帧已收手及非法输出不会触发动作；过期决策也不会因近距离物体而放行。
全部 Skills 仍保留，通用视觉策略可用 `--vision-task general` 切回。
当前远端约束解码出现 `Unexpected empty grammar stack`，因此脚本使用
`--vision-json-mode json`，请求通用 JSON 并由本地严格校验字段；不会修补输出来触发动作。
旧参数 `prompt` 为 `json` 的兼容别名。通用 JSON 也可能遇到服务端错误，并非稳定性保证。
2026-09-06 对同一批 23 个人工核对窗口做离线对照：旧 3B+原提示词的动作选择正确
12/23，新 9B+第一视角提示词正确 22/23；其中握手分别为 0/10 和 10/10，
12 个无动作窗口均没有动作误触发。新版仍将唯一挥手样本判成击掌。
随后针对挥手修正第一视角提示：`directed_at_robot` 表示招呼对象是摄像头所在机器人，
不是要求挥手时手臂直指镜头；同时区分摆动的挥手与静止接触邀请的击掌。
同一回归集复测为 23/23（握手 10、无动作 12、挥手 1），结果见
`debug/vision/eval-qwen35-wave-recipient.jsonl`。该集已参与修正验证，不能当作独立测试集；
仅有一个挥手窗口，仍需现场验证更多动作，不代表百分之百准确。
未确认的日志会列出 `recipient_unconfirmed`、`hand_not_visible`、`gesture_not_current`
或 `evidence_mismatch`；这些条件仍然会阻止执行。
这是同一人、同一房间、时间相关的回放集，不代表通用准确率，也没有验证清晰击掌正样本。
原始回放结果位于 `debug/vision/eval-qwen35-egocentric.jsonl` 和
`debug/vision/eval-qwen25-legacy-retry.jsonl`。回放中位耗时约 1.21 秒，可能受缓存影响。
首次加载较慢，过期动作仍会被拦截，不能因为首次请求慢就提高安全时效上限。

需要回退旧模型时（不改其它配置文件）：

```bash
sh scripts/run-remote-vision.sh --model qwen2.5vl:3b \
  --vision-social-profile legacy --no-vision-disable-thinking
```

离线评测通过真实 `SocialVisionAgent` 生成决策，但不调用机器人或执行 Skill：

```bash
.venv/bin/python scripts/eval-social-vision.py debug/vision/eval-heldout.json \
  debug/vision/eval-new-run.jsonl --model qwen3.5:9b --no-think
```

评测输出文件必须不存在，避免覆盖旧结果。抓帧和标签是本地隐私数据，不提交到仓库。
日志 `model_metrics.round_trip_s` 包括网络耗时；`prompt_eval_s`、`eval_s` 是服务端
输入处理与生成耗时。首次加载及相同图片的缓存命中耗时不能代表连续视频性能。
脚本没有保存 SSH 密码；隧道断开时需重新连接，服务不会自动切回本地推理。
远端推理的基准方法、当前 3 帧默认值和单/双并发对照见
[`docs/vision-latency.md`](docs/vision-latency.md)。

D435i 通过 USB 直接连接运行本程序的 Linux 主机。相机取流使用
`pyrealsense2`，不经过 Unitree SDK；Unitree bindings 负责 Go2 动作。

安装可选视觉依赖：

```bash
uv sync --extra perception
```

本地 Hugging Face 视频策略还需要：

```bash
uv sync --extra perception --extra vision
```

`vision` extra 安装 Transformers、Pillow 和 SmolVLM processor 的轻量依赖，但不会
替换 Jetson 的 CUDA PyTorch。当前 Jetson 保持 JetPack 5.1.1 / CUDA 11.4，不需要
升级系统 CUDA。默认 `cuda` 后端由主程序启动常驻 `/usr/bin/python3` 子进程，复用
系统已有的 `torch 2.0.0+nv23.05` CUDA 环境；主程序与 CUDA worker 之间只传视频帧
和 JSON 决策。CUDA 不可用时会直接报错，不会静默退回 CPU。

JetPack 5 的系统 glibc 较旧，PyPI 的 ARM64 `pyrealsense2` wheel 可能无法加载。
在 ARM64 上程序会先尝试当前 Python；如果 binding 不可用，会自动启动常驻的
`/usr/bin/python3` 相机 worker，复用 JetPack 系统中安装的 librealsense、NumPy 和
OpenCV，不会为每帧重启进程。先验证系统 Python 环境：

```bash
/usr/bin/python3 -c 'import pyrealsense2, numpy, cv2; print("RealSense ready")'
```

如果 binding 安装在其他解释器中，可传入 `--camera-python /path/to/python`，或设置
`GO2_REALSENSE_PYTHON`。不要为此升级 JetPack 5 的系统 glibc。

默认视频模型是：

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

先用 Hugging Face CLI 下载模型：

```bash
HF_HUB_DISABLE_XET=1 hf download \
  HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --exclude 'onnx/*'
```

只有一台 D435i 时无需传 `--camera-serial`。先使用模拟机器人验证真实摄像头和
CUDA VLM，不会连接或驱动实体 Go2：

```bash
.venv/bin/python -m app.perception --once --no-audio
.venv/bin/python -m app.perception --no-audio
```

上面默认等价于 `--policy vision --vision-backend cuda --vision-frame-count 1`。
当前设备的真实 8 帧测量约为 9–12 秒/次、峰值显存约 1.9 GB，因此无法用于
实时动作响应；2 帧实测仍约为 5.5–7.5 秒。`--vision-interval-s 0.5` 只是最短
调度间隔，因此当前硬件默认使用最新单帧，并拒绝执行基于超过 5 秒旧画面的动作：

```bash
.venv/bin/python -m app.perception \
  --no-audio \
  --vision-frame-count 1
```

需要保留两帧时可显式传 `--vision-frame-count 2`，但会增加延迟。
Go2 动作使用 `mobile_base`，深度安全锁阻止相应动作；`stop` / `stop_move` 仍可调用。

完成现场安全检查后，才显式增加真机参数：

```bash
.venv/bin/python -m app.perception \
  --hardware \
  --network eth0 \
  --no-audio
```

如果使用 Ollama 的 Qwen2.5-VL：

```bash
ollama serve
ollama pull qwen2.5vl:3b

uv run --extra perception go2-perception \
  --hardware \
  --network eth0 \
  --camera-serial <front-camera-serial> \
  --vision-backend ollama \
  --model qwen2.5vl:3b \
  --no-audio
```

视频决策沿用统一的 `AgentDecision` JSON，并增加两个控制动作：

```text
continue  -> 保持当前行为，不启动新 Skill
interrupt -> 取消当前可中断 Skill，并调用机器人软件 stop
```

相同 Skill/参数/语音正在执行时不会重复启动；执行结束后默认还有 5 秒冷却，可用
`--action-cooldown-s` 调整。旧事件策略仍可运行：

```bash
uv run --extra perception go2-perception \
  --policy event \
  --model qwen3:1.7b
```

事件策略处理三种稀疏事件：

```text
person_entered   -> Decision Agent -> wave / speech / ignore
person_left      -> Decision Agent -> normally ignore
person_too_close -> Decision Agent -> move_backward / speech / ignore
```

持续可见的同一个人不会重复产生 `person_entered`。默认距离小于等于 `0.8` 米时
产生一次 `person_too_close`，恢复到 `1.0` 米后才允许再次触发，可以分别使用
`--too-close-m` 和 `--too-close-release-m` 调整。

所有运行日志使用固定 JSON Lines envelope：

```json
{"schema":"go2agent.log.v1","timestamp":"...","level":"info","type":"vision_decision","owner":"agent.vision_policy","data":{}}
```

每一行都会显示 `owner`。启动日志的 `data.owners` 会列出完整 owner 清单；主要值为
`perception.camera`、`perception.realsense`、`perception.detector`、`perception.safety`、
`agent.vision_policy`、`runtime.skill` 和 `robot.adapter`。观测日志默认只打印首次结果、
状态变化和每 5 秒一条心跳；`--verbose-observations` 恢复逐帧输出，
`--observation-interval-s <seconds>` 可调整心跳间隔。

连接真实 Go2 并使用旧事件策略：

```bash
uv run --extra perception go2-perception \
  --policy event \
  --hardware \
  --network eth0
```

Decision Agent 的 `speech` 通过主机 TTS 播放；
`--no-audio` 可以关闭。动作决策仍先经过 Pydantic `AgentDecision` 校验，然后只调用
`SkillRuntime.execute()`，不会让模型直接访问 Unitree SDK。

Decision Agent 的技能目录由当前 `SkillRegistry` 动态生成，新增 Skill 后不需要再
维护另一份硬编码的技能白名单；每个动作的参数仍由对应 Skill 的 `SkillArgs` 校验。

多台 RealSense 同时连接时可增加 `--camera-serial <serial>`。默认读取
`640x480@30 FPS` 的彩色和深度流，将深度对齐到彩色画面，并忽略有效深度超过
`4` 米的检测。当前第一版使用 OpenCV HOG 全身检测器，适合验证闭环；实际场地
仍需根据视角、光照和人员距离调整阈值并做真机验收。

`move_backward` 将距离限制在 `0.05 <= distance_m <= 0.3` 米，使用 `0.1` 到
`0.3` 米/秒的速度，约每 20 ms 刷新指令；结束、异常、超时或取消均请求停止。
当前距离来自速度乘时间的开环估计，不是定位系统提供的精确位移。

## 类型检查与测试

仓库根目录的 `pyrightconfig.json` 把本目录设为 `standard` 等级，并指向项目
`.venv`，避免编辑器使用错误解释器造成第三方类型缺失或 `create_agent` 的严格
Unknown 诊断。

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check src tests
pyright src tests
uv build
```
