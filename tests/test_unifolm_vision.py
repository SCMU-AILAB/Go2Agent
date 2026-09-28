from __future__ import annotations

import json
import unittest
from typing import Self
from unittest.mock import AsyncMock, patch

from agent import UnifolmDecisionInvoker, UnifolmVisionInvoker
from agent.decision import DecisionAgentError


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class UnifolmVisionInvokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_and_invoke_preserve_server_metrics(self) -> None:
        responses = [
            FakeResponse(
                {
                    "status": "ready",
                    "model": "/models/UnifoLM-ER-1",
                    "device": "cuda:0",
                }
            ),
            FakeResponse(
                {
                    "request_id": 1,
                    "output": '{"gesture":"none"}',
                    "metrics": {"inference_s": 0.61, "generated_tokens": 12},
                }
            ),
        ]
        with patch(
            "agent.unifolm_vision.urllib.request.urlopen", side_effect=responses
        ) as urlopen:
            invoker = UnifolmVisionInvoker(
                base_url="http://127.0.0.1:8011", timeout_s=2
            )
            await invoker.warmup()
            output = await invoker.ainvoke([b"jpeg"], "runtime prompt")
            await invoker.close()

        self.assertEqual(output, '{"gesture":"none"}')
        self.assertEqual(invoker.backend_info["device"], "cuda:0")
        self.assertEqual(invoker.last_metrics["inference_s"], 0.61)
        self.assertEqual(invoker.last_metrics["frame_count"], 1)
        self.assertEqual(invoker.last_metrics["image_bytes"], 4)
        self.assertEqual(urlopen.call_count, 2)

    async def test_mismatched_request_id_is_rejected(self) -> None:
        response = FakeResponse(
            {"request_id": 99, "output": '{"gesture":"none"}', "metrics": {}}
        )
        with patch(
            "agent.unifolm_vision.urllib.request.urlopen", return_value=response
        ):
            invoker = UnifolmVisionInvoker(timeout_s=2)
            with self.assertRaisesRegex(Exception, "mismatched request id"):
                await invoker.ainvoke([b"jpeg"], "runtime prompt")

    async def test_decision_uses_visual_observation_without_forwarding_images(
        self,
    ) -> None:
        observer = AsyncMock()
        observer.ainvoke.return_value = "No person is visible."
        observer.last_metrics = {"inference_s": 0.6}
        observer.backend_info = {"status": "ready"}
        decider = AsyncMock()
        decider.ainvoke.return_value = '{"action":"ignore"}'
        decider.last_metrics = {"round_trip_s": 1.1}
        invoker = UnifolmDecisionInvoker(
            task="greet waving people",
            observer=observer,
            decision_invoker=decider,
        )

        await invoker.warmup()
        result = await invoker.ainvoke([b"frame"], "full skill catalog")
        await invoker.close()

        self.assertEqual(result, '{"action":"ignore"}')
        self.assertIn("greet waving people", observer.ainvoke.call_args.args[1])
        self.assertEqual(decider.ainvoke.call_args.args[0], ())
        self.assertIn("No person is visible", decider.ainvoke.call_args.args[1])
        self.assertIn("full skill catalog", decider.ainvoke.call_args.args[1])
        self.assertEqual(invoker.last_metrics["unifolm_metrics"], {"inference_s": 0.6})

    async def test_decision_rejects_point_output_before_tool_selection(self) -> None:
        observer = AsyncMock()
        observer.ainvoke.return_value = '```json\n[{"point":[436,573]}]\n```'
        decider = AsyncMock()
        invoker = UnifolmDecisionInvoker(
            observer=observer,
            decision_invoker=decider,
        )

        with self.assertRaisesRegex(DecisionAgentError, "unusable visual observation"):
            await invoker.ainvoke([b"frame"], "full skill catalog")

        decider.ainvoke.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
