"""The route evaluation tool measures decisions and never runs skills."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from agent.social_vision import SocialVisionAgent
from skills import build_go2_autonomy_skills


class VisualTaskEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_recorded_case_summary_reports_correct_and_stale_decisions(self):
        path = Path(__file__).parents[1] / "scripts/eval-visual-tasks.py"
        spec = importlib.util.spec_from_file_location("visual_task_eval", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        invoker = AsyncMock()
        invoker.ainvoke.return_value = {
            "action": "execute_skill",
            "skill": "follow_person",
        }
        invoker.last_metrics = {}
        cases = [
            {
                "id": "follow",
                "instruction": "跟着人走",
                "frames": ["frame.jpg"],
                "perception": {
                    "person_count": 1,
                    "nearest_person_distance_m": 2.0,
                    "person_center_x": 0.5,
                },
                "obstacle_m": 2.0,
                "expected_action": "execute_skill",
                "expected_skill": "follow_person",
                "catalog": build_go2_autonomy_skills(),
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frame.jpg").write_bytes(b"fake recorded frame")
            report = await module.evaluate(
                cases,
                lambda task: SocialVisionAgent(
                    invoker=invoker, operator_instruction=task
                ),
                root=root,
                max_age_s=0,
            )
        self.assertEqual(report["correct_count"], 1)
        self.assertEqual(report["stale_count"], 1)
        self.assertGreaterEqual(report["decision_p95_s"], report["decision_p50_s"])
        invoker.close.assert_awaited_once()
