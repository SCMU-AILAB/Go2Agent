# Normality 阶段 1–4 实施与验收记录

## 已实现

| 阶段 | 实现内容 | 验证证据 |
| --- | --- | --- |
| 1：通用任务入口 | `vision` 模式，兼容 `gesture`；用户任务与系统示例分离；模型生成任务规格；动态技能/参数校验；自主任务过滤操作员专用能力；不支持的任务显示能力缺口 | `tests/test_visual_tasks.py`：三类目标共用入口、跟随任务拒绝漂移到挥手、未知/越权技能与非法参数拒绝、能力缺口不执行 |
| 2：人员感知与跟随 | 人员横向位置与深度从相机传到运行时；新增可配置 `follow_person`；本地控制独立于模型；失去人员、歧义、过期、障碍、遥测异常、取消时停止 | `tests/test_go2_follow.py`：转向、距离控制、数据失效、目标跳变、障碍、取消、遥测；`test_visual_tasks.py`：自定义距离、模型忙时持续控制 |
| 3：闭环与时效 | 单个在途推理、最新结果队列、执行前时效复检、持续技能不重复启动、事件状态去重、取消后拒绝迟到结果、替换任务前停止旧执行、429/截断恢复；模型客户端释放 | `tests/test_vision_policy.py`、`tests/test_console_vision.py`、`tests/test_visual_tasks.py`：陈旧决策不执行、活动技能/结果反馈、模型恢复、替换、一次回应与重新触发、停止失败不宣称停止 |
| 3：共享模型与路线评估 | 服务端有界串行队列与 Retry-After；取消请求不提前释放推理锁；默认语义证据 + JSON 决策，提供直接 JSON 路线与录制帧评估工具 | `tests/test_shared_vision_server.py`：两客户端、队列超时与恢复、过载、取消；`tests/test_visual_task_evaluation.py`：结果匹配、过期计数、延迟分位数；实际模型质量比较预留如下 |
| 4：前端 | 通用视觉任务、跟随/挥手/人员出现示例；目标、距离参数、人员数、有效深度、跟随数据年龄、模型耗时、当前技能、等待/阻塞/失败原因；停止及应用新目标；确认时长 UI/API 删除 | `frontend/test/features/console/visual_task_status_test.dart`：手机布局与状态、目标替换；现有控制器/桌面/手机布局回归；Flutter 静态检查和 Web 构建 |

Python 完整回归：276 项通过。Flutter 完整回归：17 项通过。`flutter analyze` 无问题，`flutter build web` 成功。本次验证全部使用模拟机器人、假模型响应、合成画面或本地静态构建，没有连接、部署、重启或移动真实机器狗。

验证命令：

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
cd frontend
flutter analyze
flutter test
flutter build web
```

本次生产改动、新增测试和评估脚本通过 Ruff 检查；`git diff --check` 通过。API、视觉 CLI 和评估脚本的 `--help` 入口已检查。

## 运行契约

- 跟随默认距离 1.5 米；可配置范围 0.8–4.0 米。速度上限默认 0.3 m/s，可配置范围 0.2–0.5 m/s。参数在任务规格与技能执行边界验证。
- 单人可见距离跟随。多人、目标丢失、无有效深度等情形不猜测目标；不包含身份识别和绕角路径追踪。
- 控制目标周期 20 ms，人员感知有效期 500 ms；短暂检测漏帧仅使用受时效约束的 300 ms 宽限。实际调度频率与物理停止时间尚未在硬件测量。
- 视觉决策有效期默认 2 秒，执行前复检。该限制丢弃陈旧结果，不能保证模型识别正确或消除有效期内的推理延迟。
- 一次性动作按事件回应；明确条件解除后重新触发，不使用五秒冷却或手势确认时长。持续技能保持同一实例；`while_needed` 任务允许根据新证据继续有界步骤。
- 单个任务的结构化规格确定相关技能；运动任务保留站立准备和停止能力。未知技能、与任务无关的技能、非法参数、操作员专用动作不会被自主视觉执行。
- 新目标使用 API `replaceExisting=true`，先取消并停止旧执行，再生成新任务 ID。SDK 停止失败返回 502 并显示失败，不能报告已停止。
- API 相机与视觉状态展示来自真实后端契约；测试数据不代表真机识别准确率。

## 推理路线选择

默认 `grounded`：UnifoLM 描述任务相关的当前证据，文本模型输出 schema 约束 JSON。该路线复用现有模型分工，便于检查证据、参数和错误。通用提示词不再只询问手势，人员出现触发的问候也不要求挥手。

保留 `--vision-decision-route direct` 进行对照测试，直接让 UnifoLM 输出通用决策 JSON。尚没有代表性的离线人员/手势录制集，不能用假模型测试判断哪条路线在实际模型上更快或更准确。这项不确定性预留，不据此关闭时效检查或宣称真机可用。

只读评估工具：`scripts/eval-visual-tasks.py`。它调用模型并记录决策，**不调用技能执行器、不连接相机、不连接机器人**。清单格式示例：

```json
[
  {
    "id": "single-person-follow",
    "instruction": "看见一个人后跟着走，保持 1.5 米距离",
    "frames": ["frames/001.jpg", "frames/002.jpg"],
    "perception": {
      "person_count": 1,
      "nearest_person_distance_m": 2.0,
      "person_center_x": 0.5
    },
    "obstacle_m": 2.0,
    "expected_action": "execute_skill",
    "expected_skill": "follow_person"
  }
]
```

运行方式（需要已存在的录制文件和模型服务）：

```sh
PYTHONPATH=src .venv/bin/python scripts/eval-visual-tasks.py cases.json \
  --vision-url http://MODEL_SERVER:8011 \
  --ollama-url http://TEXT_MODEL_SERVER:11434 \
  --route both --output comparison.json
```

报告含任务规格、模型证据/指标、正确数、陈旧数及 p50/p95 决策耗时。需分别在单客户端和共享服务负载下运行；本次没有对外部模型服务发起评估。

## 预留的真实不确定项

以下事项不阻塞阶段 1–4 的代码与模拟验收，也不能通过纯模拟确认。遵守“全程不动机器狗”，没有开展阶段 5：

1. 实际安装方向、相机视角及人员背对狗时的检测效果；现有 HOG 是否需要替换，必须依据代表性画面评估。
2. 真实模型的任务规格准确率、视觉语义准确率和两机器人共享时的延迟分布；需要录制样本和模型评估结果。
3. SDK 3104 的物理结果、命令发送到实际停止的延迟、速度及距离误差；需要单独的硬件验证，本次不能判断。

多人身份保持、沿人的路线转弯跟随、复杂步骤规划属于后续扩展，并未作为此次完成内容。距离默认值和推理默认路线已自行选定，无需用户重复确认技术方案。
