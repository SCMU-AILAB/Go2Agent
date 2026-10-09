"""General visual goals tested with synthetic frames and simulated Go2 only."""

import asyncio
import json
import unittest
import urllib.error
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent.decision import RecoverableDecisionError
from agent.social_vision import SocialVisionAgent
from agent.unifolm_vision import UnifolmDecisionInvoker, UnifolmVisionInvoker
from agent.vision_policy import VisionPolicyWorker
from agent.visual_task import VisualTaskPlanner, VisualTaskSpec
from app.api import create_app
from app.backend import BackendConfig, ConsoleBackend
from core.runtime import SkillRuntime
from perception import PerceptionResult, VideoBuffer
from robot import RobotCommandError, RobotState, SimulatedRobotAdapter
from skills import build_go2_all_skills, build_go2_autonomy_skills, register_go2_skills
from tests.test_api import fake_agent_factory
from tests.test_console_vision import Camera, wait_for
from tests.test_go2_follow import frame


class VisualTaskContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_follow_task_cannot_drift_to_registered_wave_skill(self):
        invoker = AsyncMock()
        invoker.ainvoke.return_value = {"action": "execute_skill", "skill": "wave"}
        agent = SocialVisionAgent(
            invoker=invoker, operator_instruction="看见人后跟着走"
        )
        agent.task_spec = VisualTaskSpec(goal="跟随", required_skills=["follow_person"])
        result = await agent.decide(
            [frame()],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )
        self.assertEqual(result.action, "ignore")
        self.assertIn("skill not allowed", result.reason)

    async def test_planner_keeps_goal_and_validated_distance(self):
        invoker = AsyncMock()
        invoker.ainvoke.return_value = json.dumps(
            {
                "goal": "跟随一个人",
                "required_skills": ["follow_person"],
                "skill_arguments": {"follow_person": {"target_distance_m": 2.0}},
                "repeat_policy": "persistent",
            }
        )
        spec = await VisualTaskPlanner(invoker).plan(
            "跟着人保持两米", build_go2_all_skills()
        )
        self.assertEqual(
            spec.skill_arguments["follow_person"]["target_distance_m"], 2.0
        )
        prompt = invoker.ainvoke.call_args.args[1]
        self.assertIn("跟着人保持两米", prompt)
        self.assertNotIn('"name": "front_jump"', prompt)

    async def test_unknown_operator_and_invalid_parameters_report_gap(self):
        for name, args in [
            ("pick_up_cup", {}),
            ("front_jump", {}),
            ("follow_person", {"target_distance_m": 0.1}),
        ]:
            with self.subTest(skill=name):
                invoker = AsyncMock()
                invoker.ainvoke.return_value = {
                    "goal": "测试目标",
                    "required_skills": [name],
                    "skill_arguments": {name: args},
                }
                spec = await VisualTaskPlanner(invoker).plan(
                    "测试目标", build_go2_all_skills()
                )
                self.assertFalse(spec.supported)
                self.assertTrue(spec.capability_gap)

    async def test_three_goals_share_open_decision_without_hand_gate(self):
        for instruction, skill in [
            ("跟随人员", "follow_person"),
            ("有人出现就坐下", "sit"),
            ("挥手时回应", "wave"),
        ]:
            with self.subTest(task=instruction):
                invoker = AsyncMock()
                invoker.ainvoke.return_value = {
                    "action": "execute_skill",
                    "skill": skill,
                }
                agent = SocialVisionAgent(
                    invoker=invoker,
                    operator_instruction=instruction,
                    task_context="系统示例中提到了挥手",
                )
                result = await agent.decide(
                    [frame()],
                    RobotState(hardware=False, connected=True),
                    build_go2_autonomy_skills(),
                )
                self.assertEqual(result.skill, skill)
                prompt = invoker.ainvoke.call_args.args[1]
                self.assertIn(instruction, prompt)
                self.assertNotIn("hand_visible / directed_at_robot", prompt)

    async def test_task_parameters_override_model_drift(self):
        invoker = AsyncMock()
        invoker.ainvoke.return_value = {
            "action": "execute_skill",
            "skill": "follow_person",
            "arguments": {"target_distance_m": 1.0},
        }
        agent = SocialVisionAgent(invoker=invoker, operator_instruction="跟随保持两米")
        agent.task_spec = VisualTaskSpec(
            goal="跟随保持两米",
            skill_arguments={"follow_person": {"target_distance_m": 2.0}},
        )
        result = await agent.decide(
            [frame()],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )
        self.assertEqual(result.arguments["target_distance_m"], 2.0)

    async def test_invalid_model_arguments_are_rejected_before_execution(self):
        invoker = AsyncMock()
        invoker.ainvoke.return_value = {
            "action": "execute_skill",
            "skill": "follow_person",
            "arguments": {"max_speed_m_s": 100},
        }
        agent = SocialVisionAgent(invoker=invoker, operator_instruction="跟随")
        result = await agent.decide(
            [frame()],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )
        self.assertEqual(result.action, "ignore")
        self.assertIn("invalid skill arguments", result.reason)

    async def test_presence_greeting_does_not_require_waving(self):
        observer, decider = AsyncMock(), AsyncMock()
        observer.ainvoke.return_value = "One person stands in the center, hands down."
        observer.last_metrics, decider.last_metrics = {}, {}
        decider.ainvoke.return_value = '{"action":"execute_skill","skill":"wave","gesture_state":"none","condition_met":true}'
        invoker = UnifolmDecisionInvoker(
            observer=observer, decision_invoker=decider, task="看到人出现就问好"
        )
        invoker.allow_presence_greeting = True
        result = json.loads(await invoker.ainvoke([b"fake"], "task and catalog"))
        self.assertEqual(result["skill"], "wave")
        self.assertIn("people and their positions", observer.ainvoke.call_args.args[1])

    async def test_429_is_recoverable_but_500_is_not(self):
        invoker = UnifolmVisionInvoker()
        for code in (429, 500):
            with (
                self.subTest(code=code),
                patch(
                    "urllib.request.urlopen",
                    side_effect=urllib.error.HTTPError(
                        "http://test", code, "busy", {}, None
                    ),
                ),
            ):
                if code == 429:
                    with self.assertRaises(RecoverableDecisionError):
                        invoker._request_json("POST", "/v1/vision/invoke", {})
                else:
                    with self.assertRaises(Exception) as raised:
                        invoker._request_json("POST", "/v1/vision/invoke", {})
                    self.assertNotIsInstance(raised.exception, RecoverableDecisionError)

    async def test_custom_follow_distance_controls_position(self):
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        register_go2_skills(runtime)
        runtime.registry.get("follow_person").observe_frame(frame(distance=2.0))
        task = asyncio.create_task(
            runtime.execute("follow_person", target_distance_m=2.0)
        )
        await asyncio.sleep(0.06)
        self.assertFalse(any(name == "move_velocity" for name, _ in robot.events))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(robot.events[-1], ("stop", None))

    async def test_cancelled_late_model_result_cannot_execute(self):
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        register_go2_skills(runtime)
        invoker = AsyncMock()
        entered = asyncio.Event()

        async def stubborn(frames, prompt):
            entered.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                return {"action": "execute_skill", "skill": "wave"}

        invoker.ainvoke.side_effect = stubborn
        agent = SocialVisionAgent(invoker=invoker, operator_instruction="挥手回应")
        video = VideoBuffer(window_s=1.0)
        video.push(frame())
        worker = VisionPolicyWorker(runtime, agent, video, interval_s=0.01)
        await worker.start()
        await asyncio.wait_for(entered.wait(), 1.0)
        await asyncio.wait_for(worker.stop(), 1.0)
        self.assertFalse(any(name == "loco_action" for name, _ in robot.events))

    async def test_presence_event_answers_once_then_rearms_without_timer(self):
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        register_go2_skills(runtime)
        invoker = AsyncMock()
        invoker.last_metrics = {}
        calls = 0

        async def decision(frames, prompt):
            nonlocal calls
            calls += 1
            if calls == 3:
                return {"action": "ignore", "condition_met": False}
            return {"action": "execute_skill", "skill": "wave", "condition_met": True}

        invoker.ainvoke.side_effect = decision
        agent = SocialVisionAgent(
            invoker=invoker, operator_instruction="看到人出现就打一次招呼"
        )
        agent.task_spec = VisualTaskSpec(goal="出现时问好", trigger_kind="presence")
        video = VideoBuffer(window_s=1.0)
        video.push(frame())
        worker = VisionPolicyWorker(runtime, agent, video, interval_s=0.02)
        await worker.start()
        try:
            deadline = asyncio.get_running_loop().time() + 1.0
            while calls < 6 and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            self.assertGreaterEqual(calls, 6)
            self.assertEqual(sum(name == "loco_action" for name, _ in robot.events), 2)
        finally:
            await worker.stop()


class PersonCamera(Camera):
    def capture_frame(self):
        captured = super().capture_frame()
        return replace(
            captured,
            observation=PerceptionResult(
                observed_at_s=captured.observed_at_s,
                person_count=1,
                nearest_person_distance_m=2.2,
                person_center_x=0.5,
            ),
        )


class TaskInvoker:
    def __init__(self, skill="follow_person", busy=False):
        self.last_metrics = {}
        self.skill, self.busy = skill, busy
        self.calls = 0

    async def warmup(self):
        pass

    async def close(self):
        pass

    async def ainvoke(self, frames, prompt):
        self.calls += 1
        if self.busy and self.calls > 1:
            raise RecoverableDecisionError("shared model HTTP 429")
        return {"action": "execute_skill", "skill": self.skill, "condition_met": True}


class VisualTaskApiTests(unittest.TestCase):
    def build(self, *, busy=False):
        camera = PersonCamera()
        invoker = TaskInvoker(busy=busy)
        robot = SimulatedRobotAdapter()
        backend = ConsoleBackend(
            BackendConfig(
                robot_model="go2", audio_enabled=False, camera_source="local"
            ),
            robot=robot,
            agent_factory=fake_agent_factory,
            camera_factory=lambda: camera,
            vision_agent_factory=lambda task: SocialVisionAgent(
                invoker=invoker, operator_instruction=task
            ),
        )
        return backend, camera, invoker

    def test_follow_runs_once_and_remains_independent_of_busy_model(self):
        backend, _camera, invoker = self.build(busy=True)
        with TestClient(create_app(backend=backend)) as client:
            wait_for(client, lambda s: s["camera"]["frameAvailable"])
            response = client.post(
                "/api/v1/tasks",
                json={"instruction": "跟着人保持距离", "taskMode": "vision"},
            )
            self.assertEqual(response.status_code, 202)
            snapshot = wait_for(
                client,
                lambda s: (
                    invoker.calls >= 2
                    and s["visionTask"].get("activeSkill") == "follow_person"
                ),
            )
            self.assertTrue(snapshot["busy"])
            self.assertTrue(
                any(name == "move_velocity" for name, _ in backend.robot.events)
            )
            self.assertFalse(
                any(name == "loco_action" for name, _ in backend.robot.events)
            )
            client.post("/api/v1/tasks/current/cancel", json={})
            self.assertEqual(backend.robot.events[-1], ("stop", None))

    def test_replacing_goal_cancels_old_follow_and_changes_task_id(self):
        backend, _camera, invoker = self.build()
        with TestClient(create_app(backend=backend)) as client:
            wait_for(client, lambda s: s["camera"]["frameAvailable"])
            first = client.post(
                "/api/v1/tasks", json={"instruction": "跟随人", "taskMode": "vision"}
            ).json()
            wait_for(
                client, lambda s: s["visionTask"].get("activeSkill") == "follow_person"
            )
            invoker.skill = "sit"
            second = client.post(
                "/api/v1/tasks",
                json={
                    "instruction": "看到人坐下",
                    "taskMode": "vision",
                    "replaceExisting": True,
                },
            )
            self.assertEqual(second.status_code, 202)
            self.assertNotEqual(first["taskId"], second.json()["taskId"])
            wait_for(
                client,
                lambda s: any(
                    name == "loco_action" and value[0] == "sit"
                    for name, value in backend.robot.events
                ),
            )
            self.assertIn(("stop", None), backend.robot.events)
            client.post("/api/v1/tasks/current/cancel", json={})

    def test_capability_gap_is_visible_and_never_executes(self):
        backend, _camera, invoker = self.build()
        planner = AsyncMock()
        planner.plan.return_value = VisualTaskSpec(
            goal="拿杯子", supported=False, capability_gap="没有抓取能力"
        )
        backend._vision_agent_factory = lambda task: SocialVisionAgent(
            invoker=invoker, operator_instruction=task, task_planner=planner
        )
        with TestClient(create_app(backend=backend)) as client:
            wait_for(client, lambda s: s["camera"]["frameAvailable"])
            client.post(
                "/api/v1/tasks", json={"instruction": "拿杯子", "taskMode": "vision"}
            )
            snapshot = wait_for(
                client, lambda s: s["visionTask"].get("state") == "blocked"
            )
            self.assertEqual(snapshot["visionTask"]["reason"], "没有抓取能力")
            self.assertEqual(invoker.calls, 0)
            self.assertFalse(
                any(name == "move_velocity" for name, _ in backend.robot.events)
            )
            client.post("/api/v1/tasks/current/cancel", json={})

    def test_transient_task_planning_error_recovers_before_execution(self):
        backend, _camera, invoker = self.build()
        planner = AsyncMock()
        planner.plan.side_effect = [
            RecoverableDecisionError("token budget exhausted"),
            VisualTaskSpec(
                goal="跟随",
                required_skills=["follow_person"],
                repeat_policy="persistent",
            ),
        ]
        backend._vision_agent_factory = lambda task: SocialVisionAgent(
            invoker=invoker, operator_instruction=task, task_planner=planner
        )
        with TestClient(create_app(backend=backend)) as client:
            wait_for(client, lambda s: s["camera"]["frameAvailable"])
            client.post(
                "/api/v1/tasks", json={"instruction": "跟随", "taskMode": "vision"}
            )
            wait_for(
                client, lambda s: s["visionTask"].get("activeSkill") == "follow_person"
            )
            self.assertEqual(planner.plan.await_count, 2)
            self.assertTrue(backend.busy)
            client.post("/api/v1/robot/emergency-stop", json={})
            snapshot = client.get("/api/v1/console").json()
            self.assertEqual(snapshot["visionTask"]["state"], "stopped")
            self.assertIsNone(snapshot["visionTask"]["activeSkill"])

    def test_gesture_alias_still_starts_general_visual_task(self):
        backend, _camera, _invoker = self.build()
        with TestClient(create_app(backend=backend)) as client:
            wait_for(client, lambda s: s["camera"]["frameAvailable"])
            self.assertEqual(
                client.post(
                    "/api/v1/tasks", json={"instruction": "跟随", "taskMode": "gesture"}
                ).status_code,
                202,
            )
            client.post("/api/v1/tasks/current/cancel", json={})

    def test_sdk_stop_failure_is_exposed_without_claiming_stopped(self):
        backend, _camera, _invoker = self.build()
        with TestClient(create_app(backend=backend)) as client:
            wait_for(client, lambda s: s["camera"]["frameAvailable"])
            client.post(
                "/api/v1/tasks", json={"instruction": "跟随", "taskMode": "vision"}
            )
            wait_for(
                client, lambda s: s["visionTask"].get("activeSkill") == "follow_person"
            )
            original_stop = backend.robot.stop

            async def failed_stop():
                raise RobotCommandError("simulated SDK timeout")

            backend.robot.stop = failed_stop
            response = client.post("/api/v1/tasks/current/cancel", json={})
            backend.robot.stop = original_stop
            self.assertEqual(response.status_code, 502)
            snapshot = client.get("/api/v1/console").json()
            self.assertEqual(snapshot["visionTask"]["state"], "failed")
            self.assertIn("simulated SDK timeout", snapshot["visionTask"]["reason"])


if __name__ == "__main__":
    unittest.main()
