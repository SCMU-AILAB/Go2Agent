# G1 机器人控制台（Flutter）

> 任务模式更新：新版控制台默认「文本指令」，开启真实相机也可发送比心、坐下等指令；
> 「持续视觉交互」需单独选择。下文 local 自动进入视觉模式的说明仅适用于未传 taskMode 的旧客户端。
> Go2 视觉 Agent 按任务提示词、画面和最近执行结果持续选择已注册技能；详见 docs/go2-adapter.md。

仓库内 FastAPI 控制面的 Flutter 客户端，支持 Web 与 macOS。

## 已实现

- 通过 REST 启动/停止会话、保存系统提示词、提交及取消任务
- 通过 WebSocket 实时接收状态、日志、心跳和摄像头帧事件
- 显示 D435i RGB 画面、机器人连接状态和通信延迟
- 展示 Agent 输出、工具调用和 SkillRuntime 执行进度
- 本地相机任务接入持续 VideoBuffer → SocialVisionAgent → SkillRuntime，返回视觉决策、抑制原因和 TTS 调用状态
- 日志等级筛选、搜索、暂停与清空
- 桌面、平板和手机响应式布局

前端不会直接调用 Unitree SDK。所有动作都经过 FastAPI、`RobotAgent` 和
`SkillRuntime`；是否连接真机由后端启动参数决定。

其中本地相机任务由 `SocialVisionAgent`/`VisionPolicyWorker` 驱动，不经过文本
`RobotAgent`。Go2 每轮将最近视频窗口、任务、技能目录、机器人状态和上次执行结果
发给视觉模型，模型可选择执行、说话、继续、打断或忽略；动作仍经过本地 SkillRuntime。
系统提示词不能注册新技能，也不能绕过未启用的 operator-only 目录。
选择模拟视频源则维持一次性文本任务，不会把模拟背景当成真实图像传给VLM。

## 项目结构

```text
lib/
  main.dart                         # 启动入口，只调用 runApp
  app.dart                          # MaterialApp 与首页装配
  core/theme/                       # 应用主题、公共颜色
  features/console/
    models/                         # 后端快照、日志、工具调用数据模型
    services/console_api.dart       # REST/WebSocket API 客户端
    controllers/console_controller.dart
                                    # 服务状态、事件流、日志与界面状态
    presentation/
      console_page.dart             # 响应式布局、快捷键、消息提示
      console_intents.dart          # 快捷键动作定义
      widgets/                      # 摄像头、配置、输入、执行、日志等面板
test/
  support/fake_console_api.dart      # 可控的 API 测试替身
  widget_test.dart                   # 桌面/手机布局与任务流程
  features/console/                 # 控制器与事件流测试
```

状态使用 Flutter 自带的 `ChangeNotifier`，页面通过 `ListenableBuilder`
订阅更新。控制器启动时读取 `/api/v1/console`，之后订阅 `/api/v1/events`，
并保留低频 REST 轮询作为断线容错。

## 运行方式

### 实时视觉前后端

先停止之前的 `run-remote-vision.sh` 相机进程，避免占用同一个 D435i。
保持云端 Ollama 服务及原 SSH 隧道运行（已有隧道时不要重复启动）：

```bash
cd ~/G1Agent
sh scripts/remote-vision-tunnel.sh
```

另一个终端启动后端，先使用模拟机器人与真实相机：

```bash
cd ~/G1Agent
.venv/bin/python -m app.api --camera-source local \
  --vision-model qwen3.5:9b --vision-url http://127.0.0.1:11435 --no-audio
```

前端在同一台主机上可运行 `flutter run -d chrome`。若前端在另一台主机，后端
需要在受信任局域网监听（`--host 0.0.0.0`），前端传入
`--dart-define=G1_API_BASE_URL=http://ROBOT_IP:8000`。这是**控制台API地址**，
不是云端Ollama地址；后者只在后端用 `--vision-url` 配置。

界面选择“本地相机”，填写交互偏好，点击“开始持续视觉交互”。仅打开相机预览不会
启动动作；提交后持续运行至“停止任务”。`vision.decide` 记录帧时间、决策、模型耗时；
`vision.outcome` 记录Skill返回值、冷却/过期拦截及 `speech_spoken`。
TTS成功返回不等于麦克风确认播报完成。模拟后端不连接真实TTS。

相机默认旋转180°，预览与模型使用相同校正后的JPEG；相机正装时传
`--vision-rotation-deg 0`。深度和HOG检测坐标仍保留原始方向。
任务期间禁止切换相机或插入手动Skill；模型报错、相机中断或帧过期会停止视觉任务，
错误显示在界面，可排除故障后重新提交。停止任务会取消worker，但不是物理急停。

实机须先完成现场检查、备好急停，再在后端命令增加 `--hardware --network eth0`，
Go2 外接麦克风和扬声器时增加 `--voice --audio-input-device pulse
--audio-output-device pulse`；若提供 Piper 中文模型，再增加 `--piper-model PATH`。
5070 Ti 专用于视觉时增加 `--voice-agent-backend local_commands`，语音动作在狗端解析，
仍经 `SkillRuntime` 执行，不会占用远程视觉推理槽。
前端 VOICE 面板可查看最新转写/回复并启停监听。不要同时运行独立视觉CLI和控制台后端。
API目前无身份认证，仅限可信网络，不能开放到公网。

### 原文本/模拟流程

先启动默认模拟后端：

```bash
cd ~/G1Agent
uv run g1-api
```

再启动前端：

```bash
cd ~/G1Agent/frontend
flutter run -d chrome
```

运行 macOS 桌面版：

```bash
flutter run -d macos
```

后端运行在机器人或另一台主机时，显式指定地址：

```bash
flutter run -d chrome \
  --dart-define=G1_API_BASE_URL=http://ROBOT_IP:8000
```

真机和 D435i 后端示例：

```bash
cd ~/G1Agent
uv run g1-api \
  --hardware \
  --network eth0 \
  --camera-source local \
  --host 0.0.0.0
```

`--host 0.0.0.0` 会把控制接口暴露到局域网，只应在可信网络中使用。

## 检查与构建

```bash
flutter analyze
flutter test
flutter build web
```

Web 发布文件位于 `build/web/`。

## 接口边界

默认 API 地址为 `http://127.0.0.1:8000`。摄像头下拉框中的“本机摄像头”指
FastAPI 所在主机通过 USB 连接的 RealSense D435i，不会申请浏览器或 macOS
客户端自身的摄像头权限。
