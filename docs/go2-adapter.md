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

这些姿态接口不接受参数。G1 的 `start`、`squat`、`zero_torque`、FSM 参数、
机械臂动作 ID、挥手、握手和击掌均明确抛出 `RobotCommandError`，不转换为
语义不同的 Go2 动作。`release_arm()` 是无副作用的空操作，使现有 `StopSkill`
在完成 `stop()` 后可以正常返回；它不会触发坐下、趴下或阻尼模式。

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
from skills.posture.g1_posture import PostureSkill, PostureSpec


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
同一进程中只应有一个 DDS 所有者，不要同时连接 G1 Adapter 或另一个独占 DDS
的组件。SDK 调用使用工作线程并串行化；取消等待不会中止原生 RPC，后续
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
现有 G1 移动技能采用“一次发送 + sleep”的开环方式，不能据此保证 Go2
移动距离或持续时长；迁移时需要实现 Go2 专用移动技能。

装配入口已支持 `--robot go2`：

```bash
# 文本 CLI
.venv/bin/python -m app.main --robot go2 --hardware --network eth0

# FastAPI 控制台
.venv/bin/python -m app.api --robot go2 --hardware --network eth0 --no-audio

# 视觉闭环
.venv/bin/python -m app.perception --robot go2 --hardware --network eth0 --no-audio
```

Go2 使用 `skills.register_go2_skills()`，不会调用 `register_g1_skills()`。
自主目录包含姿态（`stand_up` / `stand_down` / `sit` / `rise_sit` /
`balance_stand`）、原生 `hello`、视觉兼容的 `wave`（底层调用 `hello`）、
周期性刷新的移动/转向技能，以及 `stop` / `stop_move`。`damp` 与
`recovery_stand` 仅在 `--include-operator-only-skills` 时注册。

Go2 移动技能会以约 20 ms 间隔刷新 `move` 速度，并在 `finally` / `cleanup()`
中调用 `stop()`；不要依赖 `close()` 停止运动。G1 的“一次发送 + sleep”
开环移动技能不会注册到 Go2 目录。

`handshake` / `high_five` 在 Go2 目录中不存在，社交视觉会将它们判为
`gesture skill unavailable`。G1 `AudioClient` TTS 在 Go2 装配时会被禁用并
写日志。当前 bindings 中的 Go2 `VuiClient` 只提供语音开关、音量和灯光亮度设置/读取，
没有文本播报接口，因此不能作为 TTS 使用；Go2 文本播报需要另外的音频输出方案。

## 扩展动作目录

新增的默认 Skills：`stretch`、`content`、`heart`（比心）、`scrape`、
`dance1`、`dance2`、`pose`。

以下 Skills 通过 `--include-operator-only-skills` 注册：
`front_flip`、`front_jump`、`front_pounce`、`left_flip`、`back_flip`、
`hand_stand`、`free_walk`、`free_bound`、`free_jump`、`free_avoid`、
`classic_walk`、`walk_upright`、`cross_step`、`static_walk`、`trot_run`、
`economic_gait`、`switch_avoid_mode`，以及原有的 `damp` / `recovery_stand`。
该选项会把这些工具同时暴露给文本 Agent 和 API；operator-only 是启动时目录划分，
并非额外的运行时权限检查。

`pose`、`hand_stand`、`free_bound`、`free_jump`、`free_avoid`、`classic_walk`、
`walk_upright`、`cross_step` 必须显式传入 JSON 布尔值 `flag`，不接受字符串或整数。
其余上述动作不需要参数。动作资源统一为 `mobile_base`，避免命令并发争用。

```python
await runtime.execute("heart")
await runtime.execute("dance1")
await runtime.execute("pose", flag=True)
await runtime.execute("pose", flag=False)
```

比心也可通过 `POST /api/v1/skills/heart/execute`、请求体 `{"arguments":{}}` 调用。
文本 Agent 的工具目录从 Registry 自动生成；原有社交视觉分类器仍只映射其支持的
手势，注册新 Skill 不会自动扩展视觉手势识别类别。

本次补齐的是上述原生动作，不含 `euler`、`speed_level`、`switch_joystick`、
`auto_recover_set/get` 等控制/查询接口。返回成功表示 SDK 接受命令，
不代表物理动作已完成；是否支持具体动作仍取决于真机固件。

## 无硬件测试

```bash
PYTHONPATH=src python3 -m unittest tests.test_go2_adapter -v
ruff check src/robot/go2_adapter.py src/robot/__init__.py tests/test_go2_adapter.py
```

测试注入 `Go2Bindings` 替身，不会导入原生 SDK、初始化真实 DDS 或连接机器人。
