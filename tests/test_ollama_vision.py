import unittest
from unittest.mock import AsyncMock, patch

from agent.decision import DecisionAgentError
from agent.vision_policy import OllamaVisionInvoker


class OllamaVisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_preloads_weights_without_a_decision(self):
        invoker = self.invoker({})
        await invoker.warmup()
        invoker._client.generate.assert_awaited_once_with(
            model="test", prompt="", keep_alive="30m",
        )
        invoker._client.chat.assert_not_called()

    def invoker(self, payload):
        invoker = OllamaVisionInvoker("test", constrain_json=False)
        invoker._client = AsyncMock()
        invoker._client.chat.return_value = payload
        return invoker

    async def test_generic_json_is_explicit_nonstreaming(self):
        invoker = self.invoker({
            "done": True, "done_reason": "stop", "message": {"content": "{}"},
        })
        self.assertEqual(await invoker.ainvoke([b"jpeg"], "test"), "{}")
        kwargs = invoker._client.chat.call_args.kwargs
        self.assertEqual(kwargs["format"], "json")
        self.assertIs(kwargs["stream"], False)
        self.assertNotIn("think", kwargs)

    async def test_thinking_can_be_explicitly_disabled(self):
        invoker = self.invoker({"done": True, "message": {"content": "{}"}})
        invoker.think = False
        await invoker.ainvoke([b"jpeg"], "test")
        self.assertIs(invoker._client.chat.call_args.kwargs["think"], False)

    async def test_metrics_separate_encoding_http_and_server_time(self):
        invoker = self.invoker(
            {
                "done": True,
                "done_reason": "stop",
                "message": {"content": "{}"},
                "total_duration": 400_000_000,
                "prompt_eval_duration": 250_000_000,
                "eval_duration": 100_000_000,
            }
        )

        with patch(
            "agent.vision_policy.time.monotonic",
            side_effect=(10.0, 10.01, 10.51),
        ):
            await invoker.ainvoke([b"jpeg-1", b"jpeg-2"], "test")

        self.assertEqual(invoker.last_metrics["client_encode_s"], 0.01)
        self.assertEqual(invoker.last_metrics["http_round_trip_s"], 0.5)
        self.assertEqual(invoker.last_metrics["round_trip_s"], 0.51)
        self.assertEqual(invoker.last_metrics["transport_residual_s"], 0.1)
        self.assertEqual(invoker.last_metrics["network_rtt_s"], 0.1)
        self.assertEqual(invoker.last_metrics["image_bytes"], 12)

    async def test_incomplete_and_token_limited_responses_rejected(self):
        for payload in (
            {"done": False}, {},
            {"done": True, "done_reason": "length", "message": {"content": "{}"}},
        ):
            with self.subTest(payload=payload), self.assertRaises(DecisionAgentError):
                await self.invoker(payload).ainvoke([b"jpeg"], "test")
