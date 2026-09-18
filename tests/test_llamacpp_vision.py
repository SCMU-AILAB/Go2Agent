from __future__ import annotations

import json
import unittest
from typing import Self
from unittest.mock import patch

from agent import LlamaCppVisionInvoker
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


class LlamaCppVisionInvokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_and_multiframe_completion(self) -> None:
        responses = [
            FakeResponse({"status": "ok"}),
            FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": '{"action":"ignore"}'},
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 5},
                    "timings": {"prompt_ms": 400.0, "predicted_ms": 100.0},
                }
            ),
        ]
        with patch(
            "agent.llamacpp_vision.urllib.request.urlopen", side_effect=responses
        ) as urlopen:
            invoker = LlamaCppVisionInvoker(timeout_s=2)
            await invoker.warmup()
            output = await invoker.ainvoke([b"jpeg-1", b"jpeg-2"], "prompt")

        self.assertEqual(output, '{"action":"ignore"}')
        self.assertEqual(invoker.backend_info["backend"], "llama.cpp")
        self.assertEqual(invoker.last_metrics["inference_s"], 0.5)
        self.assertEqual(invoker.last_metrics["input_tokens"], 100)
        self.assertEqual(invoker.last_metrics["frame_count"], 2)
        request = urlopen.call_args_list[1].args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:8012/v1/chat/completions")
        response_format = body["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "vision_output")
        self.assertIs(response_format["json_schema"]["strict"], True)
        schema = response_format["json_schema"]["schema"]
        self.assertIn("action", schema["properties"])
        self.assertIn("action", schema["required"])
        self.assertEqual(len(body["messages"][0]["content"]), 3)

    async def test_custom_output_schema_is_forwarded(self) -> None:
        output_schema = {
            "type": "object",
            "properties": {"gesture": {"const": "none"}},
            "required": ["gesture"],
            "additionalProperties": False,
        }
        response = FakeResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"gesture":"none"}'},
                    }
                ]
            }
        )
        with patch(
            "agent.llamacpp_vision.urllib.request.urlopen", return_value=response
        ) as urlopen:
            output = await LlamaCppVisionInvoker(
                timeout_s=2,
                output_schema=output_schema,
            ).ainvoke([b"jpeg"], "prompt")

        self.assertEqual(output, '{"gesture":"none"}')
        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(
            body["response_format"]["json_schema"]["schema"],
            output_schema,
        )

    async def test_token_limited_completion_is_rejected(self) -> None:
        response = FakeResponse(
            {
                "choices": [
                    {"finish_reason": "length", "message": {"content": "{}"}}
                ]
            }
        )
        with patch(
            "agent.llamacpp_vision.urllib.request.urlopen", return_value=response
        ), self.assertRaisesRegex(DecisionAgentError, "token budget"):
            await LlamaCppVisionInvoker(timeout_s=2).ainvoke([b"jpeg"], "prompt")


if __name__ == "__main__":
    unittest.main()
