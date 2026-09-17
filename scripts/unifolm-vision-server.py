#!/usr/bin/env python3
"""Resident HTTP service for Unitree UnifoLM-ER-1 on a CUDA server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import time
from contextlib import asynccontextmanager
from threading import Lock

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class InvokeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: int
    model: str
    frames: list[str] = Field(min_length=1, max_length=8)
    prompt: str = Field(min_length=1, max_length=32_000)
    max_new_tokens: int = Field(default=96, ge=1, le=256)


class UnifolmRuntime:
    def __init__(self, model_path: str) -> None:
        loaded_at = time.perf_counter()
        # The published tokenizer config contains a legacy list-form metadata
        # field. All action tokens already exist in tokenizer.json.
        self.processor = AutoProcessor.from_pretrained(
            model_path, extra_special_tokens={}
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()
        torch.cuda.synchronize()
        self.model_path = model_path
        self.load_s = round(time.perf_counter() - loaded_at, 4)
        self._lock = Lock()

    def invoke(self, body: InvokeRequest) -> dict[str, object]:
        started = time.perf_counter()
        try:
            images = [
                Image.open(io.BytesIO(base64.b64decode(value, validate=True))).convert(
                    "RGB"
                )
                for value in body.frames
            ]
        except Exception as exc:
            raise ValueError(f"invalid JPEG frame: {exc}") from exc
        content: list[dict[str, object]] = [
            {"type": "image", "image": image} for image in images
        ]
        content.append({"type": "text", "text": body.prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        prepared_at = time.perf_counter()
        with self._lock, torch.inference_mode():
            torch.cuda.synchronize()
            inference_started = time.perf_counter()
            generated = self.model.generate(
                **inputs,
                max_new_tokens=body.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            torch.cuda.synchronize()
            inference_finished = time.perf_counter()
        trimmed = generated[:, inputs["input_ids"].shape[-1] :]
        output = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        finished = time.perf_counter()
        return {
            "request_id": body.request_id,
            "output": output,
            "metrics": {
                "preprocess_s": round(prepared_at - started, 4),
                "inference_s": round(inference_finished - inference_started, 4),
                "decode_s": round(finished - inference_finished, 4),
                "server_total_s": round(finished - started, 4),
                "input_tokens": int(inputs["input_ids"].shape[-1]),
                "generated_tokens": int(trimmed.shape[-1]),
                "frame_count": len(images),
            },
        }


def create_app(model_path: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = await asyncio.to_thread(UnifolmRuntime, model_path)
        app.state.inference_lock = asyncio.Lock()
        yield

    app = FastAPI(title="UnifoLM Vision Server", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, object]:
        runtime: UnifolmRuntime = app.state.runtime
        return {
            "status": "ready",
            "model": runtime.model_path,
            "device": str(runtime.model.device),
            "dtype": str(runtime.model.dtype),
            "load_s": runtime.load_s,
        }

    @app.post("/v1/vision/invoke")
    async def invoke(body: InvokeRequest) -> dict[str, object]:
        runtime: UnifolmRuntime = app.state.runtime
        inference_lock: asyncio.Lock = app.state.inference_lock
        if inference_lock.locked():
            raise HTTPException(
                status_code=429,
                detail="inference already in flight; submit the latest window later",
            )
        try:
            async with inference_lock:
                return await asyncio.to_thread(runtime.invoke, body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/qwq/models/UnifoLM-ER-1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.model), host=args.host, port=args.port, log_level="info"
    )


if __name__ == "__main__":
    main()
