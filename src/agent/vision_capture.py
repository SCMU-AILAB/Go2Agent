"""Opt-in, bounded local capture of model inputs and policy results."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


class VisionCapture:
    def __init__(self, directory: Path, limit: int = 20):
        if limit <= 0:
            raise ValueError("capture limit must be positive")
        self.directory = directory.resolve()
        self.limit = limit
        self.count = 0
        self.session = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]

    @property
    def available(self):
        return self.count < self.limit

    def begin(self, frames):
        if not self.available:
            return None
        self.count += 1
        folder = self.directory / self.session / f"window-{self.count:04d}"
        folder.mkdir(parents=True, mode=0o700, exist_ok=False)
        entries = []
        for index, frame in enumerate(frames):
            data = bytes(frame.rgb)
            filename = f"frame-{index:02d}.jpg"
            (folder / filename).write_bytes(data)
            entries.append({"file": filename, "observed_at_s": frame.observed_at_s,
                            "sha256": hashlib.sha256(data).hexdigest()})
        self._write(folder / "input.json", {
            "schema": "go2agent.vision_capture.v1", "frames": entries,
            "note": "Exact JPEG inputs sent to Ollama, chronological order; not action execution evidence",
        })
        return folder

    def finish(self, folder, result):
        self._write(folder / "result.json", result)

    @staticmethod
    def _write(path, payload):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
