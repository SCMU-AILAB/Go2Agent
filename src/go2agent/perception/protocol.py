"""Wire protocol between the dog-side tracker client and the GPU-server perception service.

Transport: one WebSocket at ws://<server>:<port>/track.
Client -> server, alternating JSON text + binary:
  1. text  FrameHeader JSON
  2. binary JPEG bytes of the RGB frame
Client -> server control (text only): AddBox JSON
Server -> client (text only): TracksMsg JSON per processed frame, AddBoxResult for AddBox.
The server processes only the newest frame; it may skip frames under load.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class FrameHeader(BaseModel):
    type: Literal["frame"] = "frame"
    seq: int
    stamp: float


class AddBox(BaseModel):
    type: Literal["add_box"] = "add_box"
    request_id: int
    frame_seq: int  # must be one of the last few frames the server received
    box: tuple[float, float, float, float]  # normalized xyxy
    label: str


class TrackMsg(BaseModel):
    id: int
    label: str
    box: tuple[float, float, float, float]  # normalized xyxy
    score: float


class TracksMsg(BaseModel):
    type: Literal["tracks"] = "tracks"
    seq: int
    stamp: float
    items: list[TrackMsg]
    infer_ms: float


class AddBoxResult(BaseModel):
    type: Literal["add_box_result"] = "add_box_result"
    request_id: int
    track_id: int | None
    error: str | None = None
