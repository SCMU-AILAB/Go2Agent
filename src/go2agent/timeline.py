"""Run log: JSONL timeline plus one replayable JSON file (with JPEGs) per brain think.

Layout of a run directory:
    timeline.jsonl            one record per line: {"wall", "mono", "kind", "data"}
    thinks/0001.json          the exact chat messages sent to the VLM (images replaced by file names) + reply
    thinks/0001-0.jpg ...     the images of that think
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DATA_PREFIX = "data:image/jpeg;base64,"


class Timeline:
    def __init__(self, root: Path | None) -> None:
        """`root=None` disables writing (records are still returned for live subscribers)."""
        self.dir = root / time.strftime("%Y%m%d-%H%M%S") if root else None
        self._lock = threading.Lock()
        self._think_n = 0
        self._file = None
        if self.dir:
            (self.dir / "thinks").mkdir(parents=True, exist_ok=True)
            self._file = (self.dir / "timeline.jsonl").open("a", encoding="utf-8")
            log.info("logging run to %s", self.dir)

    def record(self, kind: str, data: dict[str, Any], mono: float | None = None) -> dict[str, Any]:
        rec = {
            "wall": time.time(),
            "mono": time.monotonic() if mono is None else mono,
            "kind": kind,
            "data": data,
        }
        if self._file:
            line = json.dumps(rec, ensure_ascii=False, default=str)
            with self._lock:
                self._file.write(line + "\n")
                self._file.flush()
        return rec

    def save_think(self, messages: list[dict[str, Any]], reply: dict[str, Any]) -> str | None:
        """Persist one think for replay. Returns its id (e.g. '0007') or None when disabled."""
        if not self.dir:
            return None
        with self._lock:
            self._think_n += 1
            tid = f"{self._think_n:04d}"
        stripped = json.loads(json.dumps(messages))
        n = 0
        for m in stripped:
            if not isinstance(m.get("content"), list):
                continue
            for part in m["content"]:
                url = part.get("image_url", {}).get("url", "")
                if url.startswith(_DATA_PREFIX):
                    name = f"{tid}-{n}.jpg"
                    (self.dir / "thinks" / name).write_bytes(base64.b64decode(url[len(_DATA_PREFIX) :]))
                    part["image_url"]["url"] = name
                    n += 1
        path = self.dir / "thinks" / f"{tid}.json"
        path.write_text(json.dumps({"messages": stripped, "reply": reply}, ensure_ascii=False, indent=1))
        return tid

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


def load_think(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inverse of save_think: messages with images re-inlined, plus the recorded reply."""
    doc = json.loads(path.read_text())
    for m in doc["messages"]:
        if not isinstance(m.get("content"), list):
            continue
        for part in m["content"]:
            url = part.get("image_url", {}).get("url")
            if url and not url.startswith("data:"):
                part["image_url"]["url"] = (
                    _DATA_PREFIX + base64.b64encode((path.parent / url).read_bytes()).decode()
                )
    return doc["messages"], doc["reply"]
