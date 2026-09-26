# Go2 Adapter

`robot.UnitreeGo2Adapter` 实现现有 `RobotAdapter` 协议，使用同级项目
`unitree_sdk2_bindings` 提供的 `unitree_sdk2_cpp.robot.go2.SportClient`。
SDK 在 `connect()` 时延迟导入；无 SDK 的开发机可以导入模块并运行替身测试。

## 已支持的操作

| Adapter 调用 | Go2 SDK 调用 |
| --- | --- |
| `move_velocity(vx, vy, yaw)` | `SportClient.move(vx, vy, yaw)` |
| `stop()` | `SportClient.stop_move()` |
| `execute_loco_action("stand_up")` | `stand_up()`，起立 |
| `execute_loco_action("stand_down")` | `stand_down()`，趴下 |
| `execute_loco_action("balance_stand")` | `balance_stand()`，平衡站立 |
| `execute_loco_action("sit")` | `sit()`，坐下 |
| `execute_loco_action("rise_sit")` | `rise_sit()`，从坐姿起身 |
| `execute_loco_action("hello")` | `hello()`，Go2 原生打招呼 |
| `execute_loco_action("stretch")` | `stretch()`，伸展 |
| `execute_loco_action("recovery_stand")` | `recovery_stand()`，恢复站立 |
| `execute_loco_action("damp")` | `damp()`，阻尼模式 |
| `execute_loco_action("stop_move")` | `stop_move()` |

这些姿态接口不接受参数。适配器仅暴露 Go2 移动与原生动作接口，未支持动作和
多余参数抛出 `RobotCommandError`。`stop` 只调用 `stop_move()`，不改变姿态。

默认速度上限为前后/横移各 0.3 m/s、转向 0.6 rad/s，可通过配置修改。
越界、非有限数和布尔值会被拒绝，不会静默截断速度。

## 初始化与 Runtime 接入

在机器人主机安装当前项目和 bindings 后，可按以下方式组装。该示例只获取
本地初始化状态，不发送动作。

```python
import asyncio

from core.runtime import SkillRuntime
from robot import UnitreeGo2Adapter, UnitreeGo2Config
from skills.motions import StopSkill
from skills.posture import PostureSkill, PostureSpec


async def main():
    robot = UnitreeGo2Adapter(UnitreeGo2Config(network_interface="eth0"))
    try:
        await robot.connect()
        runtime = SkillRuntime(robot)
        runtime.register(StopSkill())
        runtime.register(PostureSkill(PostureSpec(
            "stand_up", "stand_up", "Stand up using the Go2 sport controller."
        )))
        print(await robot.get_state())
        # 上层接到明确动作请求后，可调用：
        # result = await runtime.execute("stand_up")
    finally:
        await robot.close()


asyncio.run(main())
```

`connect()` 仅初始化 DDS 和 SportClient；`close()` 仅释放通信资源，不发送停止。
同一进程中只应有一个 DDS 所有者，不要同时连接另一个独占 DDS 的组件。SDK 调用使用工作线程并串行化；取消等待不会中止原生 RPC，后续
`stop()` / `close()` 会等在途调用返回，延迟受 SDK 超时和原生调用行为影响。

`get_state().connected` 表示本地客户端已初始化；Go2 Adapter 同时订阅
`rt/sportmodestate`，并在 `details` 中提供最新的 `mode`、`gait_type`、位置、速度、
姿态和 `error_code`。`telemetry_available` 只在最近 2 秒收到状态时为 true。
SDK 状态码为 0 只表示命令被接受；它仍不等价于动作物理完成，非零、无效状态或
调用异常统一转成 `RobotCommandError`。

## 持续移动与后续接入边界

`move_velocity()` 只发送一次速度指令，不启动后台循环。Go2 的持续运动技能
需要在限定时长内周期性刷新速度（本地 SDK 示例使用约 20 ms 间隔），并在
`finally` / Skill `cleanup()` 中调用 `stop()`，不能依靠 `close()` 停止运动。
Go2 专用移动技能已实现周期刷新；距离仍是速度乘时长的开环估计。

装配入口固定使用 Go2：

```bash
# 文本 CLI
.venv/bin/python -m app.main --hardware --network eth0

# FastAPI 控制台
.venv/bin/python -m app.api --hardware --network eth0 --no-audio

# 视觉闭环
.venv/bin/python -m app.perception --hardware --network eth0 --no-audio
```

Go2 使用 `skills.register_go2_skills()` 注册完整目录。
自主目录包含姿态（`stand_up` / `stand_down` / `sit` / `rise_sit` /
`balance_stand`）、原生 `hello`、视觉兼容的 `wave`（底层调用 `hello`）、
周期性刷新的移动/转向技能，以及 `stop` / `stop_move`。`damp` 与
`recovery_stand` 同样默认注册，文本和视觉 Agent 共享全部 46 个技能。

`follow_person` 是单目标视觉反馈 Skill：持续收到 D435i 单人检测、有效深度及
安全空间时维持约 1.5 m；前进速度保持在 0.20--0.30 m/s、0.3 rad/s 转向。丢失、多人、
障碍、相机过期或取消均调用 `stop()`。目前依赖 HOG 人体框中心，并非身份
跟踪器；多人交错或遮挡后不能保证继续跟随同一人，不可视为通用导航。

Go2 移动技能会以约 20 ms 间隔刷新 `move` 速度，并在 `finally` / `cleanup()`
中调用 `stop()`；不要依赖 `close()` 停止运动。

`handshake` / `high_five` 在 Go2 目录中不存在，社交视觉会将它们判为
未注册技能。当前 bindings 中的
`VuiClient` 只提供语音开关、音量和灯光亮度设置/读取，没有文本播报接口，因此本项目
使用狗端电脑的外接麦克风/扬声器：常驻 Faster Whisper 负责 STT，Piper（或
`espeak-ng` 回退）负责 TTS。FastAPI 增加 `--voice` 后自动监听，也可由前端 VOICE
面板调用 `/api/v1/voice/start` 和 `/api/v1/voice/stop` 控制。户外默认增加
`--voice-agent-backend vision`，把 ASR 文本作为持续视觉目标发给 5070 Ti 的
视觉 Agent，由它结合新画面、Skill 目录和执行结果选择动作。明确停止指令在狗端
优先中断，不等待模型。无视觉服务时可改用 `local_commands` 有限指令集；
`shared` 模式则复用独立文本 Agent。

## 扩展动作目录

新增的默认 Skills：`stretch`、`content`、`heart`（比心）、`scrape`、
`dance1`、`dance2`、`pose`。

以下控制和特技 Skills 同样默认注册：
`front_flip`、`front_jump`、`front_pounce`、`left_flip`、`back_flip`、
`hand_stand`、`free_walk`、`free_bound`、`free_jump`、`free_avoid`、
`classic_walk`、`walk_upright`、`cross_step`、`static_walk`、`trot_run`、
`economic_gait`、`switch_avoid_mode`、`switch_joystick`、`auto_recover_set`，以及
`damp` / `recovery_stand`。
所有工具同时暴露给文本 Agent、视觉 Agent 和 API；`operator_only` 和 `dangerous`
标签提供动作提示。视觉不再按标签或任务关键词过滤目录，仍遵守参数、连接和深度校验。

`pose`、`hand_stand`、`free_bound`、`free_jump`、`free_avoid`、`classic_walk`、
`walk_upright`、`cross_step`、`switch_joystick`、`auto_recover_set` 必须显式传入
JSON 布尔值 `flag`，不接受字符串或整数。
其余上述动作不需要参数。动作资源统一为 `mobile_base`，避免命令并发争用。

```python
await runtime.execute("heart")
await runtime.execute("dance1")
await runtime.execute("pose", flag=True)
await runtime.execute("pose", flag=False)
```

比心也可通过 `POST /api/v1/skills/heart/execute`、请求体 `{"arguments":{}}` 调用。
文本 Agent 和 Go2 持续视觉 Agent 的工具目录都从 Registry 生成。视觉模型可以
按开放目标选择当前允许的 Skill；注册新 Skill 会进入候选目录，但是否能正确理解
目标仍取决于模型与现场验证，不能绕过本地参数、感知和安全校验。

上述目录不含 `euler`、`speed_level`、`auto_recover_get` 等控制/查询接口。返回成功表示 SDK 接受命令，
不代表物理动作已完成；是否支持具体动作仍取决于真机固件。

## 无硬件测试

```bash
PYTHONPATH=src python3 -m unittest tests.test_go2_adapter -v
ruff check src/robot/go2_adapter.py src/robot/__init__.py tests/test_go2_adapter.py
```

测试注入 `Go2Bindings` 替身，不会导入原生 SDK、初始化真实 DDS 或连接机器人。

## 文本指令与真实相机独立（2026-09-17）

控制台任务面板现在显式选择「文本指令」或「持续视觉交互」，默认文本指令。
相机来源只控制预览输入，真实相机开启时也可以发送「比心」「坐下」「跳舞」。
Go2 已注册原生 `heart`、`sit`、`dance1/dance2`，无需替换成 `hello`。
文本 Agent 不接收图像，不提供视觉导航或避障保证；移动维持原有短距限制。
Go2 持续视觉模式将文字作为目标，结合画面与执行结果选择当前可用 Skill，
不再局限于手势映射。所有注册技能均可供选择；未注册名称仍会被拒绝。

API 提交文本任务并保留相机预览：

```json
{"instruction":"给我比个心","cameraSource":"local","taskMode":"text"}
```

持续视觉任务仍用兼容字段 `taskMode: "gesture"`，必须选择 `local`。
旧客户端省略 `taskMode` 时保持历史行为（local=持续视觉，demo=文本），需要更新前端
才能看到模式选项。相机预览正常不代表已经选择了文本模式。

现场排查：检查 `/api/v1/console` 的 `robot.details.robot_model` 为 `go2`，
`/api/v1/skills` 含 `heart`；日志 `收到任务 [text/gesture]` 表示选用路径，
`tool_count=0` 表示未选工具，具体工具的 `success=false` 表示执行失败。
SDK 接受命令不等于真机动作完成；固件不支持时应显示返回错误，不替换为打招呼。

本机 Jetson 现有模型服务部署示例（文本和视觉共用现有 Ollama 隧道）：

```sh
.venv/bin/python -m app.api --hardware --network eth0 \
  --camera-source local --vision-rotation-deg 0 --no-audio \
  --model qwen3.5:9b --ollama-url http://127.0.0.1:11435
```

必须确认文本模型地址与型号可用；只配置 `--vision-url` 不会自动配置文本 Agent。

持续视觉任务示例：`{"instruction":"跟着前面的人走，保持距离","cameraSource":"local","taskMode":"gesture"}`。
Agent 根据任务与视频选择已注册 Skill，`follow_person` 在狗端使用 D435i 深度反馈；
旧字段 `waveResponse` 已忽略。

本机可用 `sh scripts/run-go2-console.sh` 重启同样配置的控制台。
脚本连接真机但不会提交动作任务；网卡默认 eth0，可用 GO2_NETWORK 覆盖。
