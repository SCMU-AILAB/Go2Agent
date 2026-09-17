import unittest

from fastapi.testclient import TestClient

from agent.social_vision import SocialVisionAgent
from app.api import create_app
from robot import RobotState
from skills.go2_catalog import build_go2_autonomy_skills
from tests import test_task_modes
from tests.test_console_vision import VisionInvoker, wait_for
from tests.test_vision_policy import FakeVisionInvoker, camera_frame


class GestureResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_heart_mapping_keeps_evidence_and_active_skill_checks(self):
        for visible, active, action in [
            (True, None, "execute_skill"),
            (False, None, "ignore"),
            (True, "heart", "continue"),
        ]:
            agent = SocialVisionAgent(
                wave_response="heart",
                invoker=FakeVisionInvoker(
                    [
                        {
                            "gesture": "wave",
                            "hand_visible": visible,
                            "directed_at_robot": True,
                            "present_in_latest": True,
                            "evidence": "side_to_side",
                        }
                    ]
                ),
            )
            decision = await agent.decide(
                [camera_frame(1), camera_frame(2)],
                RobotState(hardware=False, connected=True),
                build_go2_autonomy_skills(),
                policy_context={"active_skill": active},
            )
            self.assertEqual(decision.action, action)
            if action == "execute_skill":
                self.assertEqual(decision.skill, "heart")

    async def test_mapping_does_not_enable_unregistered_handshake(self):
        agent = SocialVisionAgent(
            wave_response="heart",
            invoker=FakeVisionInvoker(
                [
                    {
                        "gesture": "handshake",
                        "hand_visible": True,
                        "directed_at_robot": True,
                        "present_in_latest": True,
                        "evidence": "offered_hand",
                    }
                ]
            ),
        )
        decision = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )
        self.assertEqual(decision.action, "ignore")

    def test_other_skills_cannot_be_selected_as_wave_response(self):
        for skill in ["damp", "front_flip", "move_forward"]:
            with self.assertRaises(ValueError):
                SocialVisionAgent(wave_response=skill, invoker=FakeVisionInvoker([]))


class GestureResponseApiTests(unittest.TestCase):
    def test_local_wave_executes_selected_heart_and_reports_it(self):
        backend = test_task_modes.TaskModeTests().build()
        invoker = VisionInvoker()
        backend._vision_agent_factory = lambda _: SocialVisionAgent(
            invoker=invoker, generate_speech=True
        )
        with TestClient(create_app(backend=backend)) as client:
            wait_for(client, lambda s: s["camera"]["frameAvailable"])
            response = client.post(
                "/api/v1/tasks",
                json={
                    "instruction": "用户挥手时比心",
                    "taskMode": "gesture",
                    "waveResponse": "heart",
                },
            )
            self.assertEqual(response.status_code, 202)
            state = wait_for(
                client, lambda s: any(t["name"] == "heart" for t in s["tools"])
            )
            self.assertNotIn("hello", [t["name"] for t in state["tools"]])
            client.post("/api/v1/tasks/current/cancel", json={})
