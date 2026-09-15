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

`get_state().connected` 表示本地客户端已初始化，**不代表已收到 Go2 在线反馈**。
`details` 显式标记 `state_source="local_client"`、`telemetry_available=False`。
此版本没有订阅 `rt/sportmodestate`，也不提供物理动作完成验证。SDK 状态码
为 0 只表示命令被接受；非零、无效状态或调用异常统一转成 `RobotCommandError`。

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
写日志；Go2 语音需要单独的 VuiClient 适配，当前版本未提供。

## 无硬件测试

```bash
PYTHONPATH=src python3 -m unittest tests.test_go2_adapter -v
ruff check src/robot/go2_adapter.py src/robot/__init__.py tests/test_go2_adapter.py
```

测试注入 `Go2Bindings` 替身，不会导入原生 SDK、初始化真实 DDS 或连接机器人。
