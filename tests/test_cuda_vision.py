from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from agent import CudaVisionInvoker

FAKE_WORKER = """
import base64
import json
import sys

print(json.dumps({
    "type": "ready",
    "backend": {"device": "cuda:0", "dtype": "torch.float16"},
}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("command") == "invoke":
        assert base64.b64decode(request["frames"][0]) == b"jpeg"
        assert "runtime prompt" in request["prompt"]
        print(json.dumps({
            "type": "result",
            "request_id": request["request_id"],
            "output": "{\\"action\\":\\"ignore\\"}",
            "metrics": {"inference_s": 0.01},
        }), flush=True)
    elif request.get("command") == "close":
        print(json.dumps({"type": "closed"}), flush=True)
        break
"""


class CudaVisionInvokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_response_after_timeout_is_skipped(self) -> None:
        invoker = CudaVisionInvoker("fake-model", inference_timeout_s=0.01)
        with self.assertRaisesRegex(Exception, "timed out"):
            await invoker._read_response_async(
                expected_type="result", request_id=1, timeout_s=0.01
            )
        invoker._lines.put(json.dumps({"type": "result", "request_id": 1}))
        invoker._lines.put(json.dumps({"type": "result", "request_id": 2}))

        result = await invoker._read_response_async(
            expected_type="result", request_id=2, timeout_s=0.1
        )
        self.assertEqual(result["request_id"], 2)

    async def test_persistent_worker_warms_up_invokes_and_closes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "fake_cuda_worker.py"
            worker.write_text(FAKE_WORKER, encoding="utf-8")
            invoker = CudaVisionInvoker(
                "fake-model",
                python_executable=sys.executable,
                packages_path=root,
                worker_path=worker,
                startup_timeout_s=3.0,
                inference_timeout_s=3.0,
            )

            await invoker.warmup()
            first = await invoker.ainvoke([b"jpeg"], "runtime prompt")
            second = await invoker.ainvoke([b"jpeg"], "runtime prompt")
            invoker._lines.put(
                json.dumps(
                    {
                        "type": "result",
                        "request_id": 999,
                        "output": '{"action":"ignore"}',
                    }
                )
                + "\n"
            )
            await invoker.close()

        self.assertEqual(first, '{"action":"ignore"}')
        self.assertEqual(second, first)
        self.assertEqual(invoker.backend_info, {})
        self.assertFalse(invoker.opened)

    async def test_concurrent_calls_are_serialized_by_one_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "fake_cuda_worker.py"
            worker.write_text(FAKE_WORKER, encoding="utf-8")
            invoker = CudaVisionInvoker(
                "fake-model",
                python_executable=sys.executable,
                packages_path=root,
                worker_path=worker,
                startup_timeout_s=3.0,
                inference_timeout_s=3.0,
            )
            try:
                outputs = await asyncio.gather(
                    invoker.ainvoke([b"jpeg"], "runtime prompt"),
                    invoker.ainvoke([b"jpeg"], "runtime prompt"),
                )
            finally:
                await invoker.close()

        self.assertEqual(outputs, ['{"action":"ignore"}'] * 2)


if __name__ == "__main__":
    unittest.main()
