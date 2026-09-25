"""Python 3.8-compatible RealSense capture worker used by realsense_bridge."""

import argparse
import base64
import json
import math
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--detection-fps", type=float, required=True)
    parser.add_argument("--frame-timeout-ms", type=int, required=True)
    parser.add_argument("--min-score", type=float, required=True)
    parser.add_argument("--max-distance-m", type=float)
    parser.add_argument(
        "--rgb-rotation-deg", type=int, choices=(0, 90, 180, 270), default=0
    )
    return parser.parse_args()


def send(message):
    print(json.dumps(message, ensure_ascii=True), flush=True)


def confidence(score):
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    exp_score = math.exp(score)
    return exp_score / (1.0 + exp_score)


def rotate_bgr_image(cv2, image, degrees):
    if degrees == 0:
        return image
    rotation_codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return cv2.rotate(image, rotation_codes[degrees])


class Detector:
    def __init__(self, args):
        import cv2  # pyright: ignore[reportMissingImports]
        import numpy  # pyright: ignore[reportMissingImports]
        import pyrealsense2 as rs  # pyright: ignore[reportMissingImports]

        self.args = args
        self.cv2 = cv2
        self.numpy = numpy
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if args.serial:
            config.enable_device(args.serial)
        config.enable_stream(
            rs.stream.depth,
            args.width,
            args.height,
            rs.format.z16,
            args.fps,
        )
        config.enable_stream(
            rs.stream.color,
            args.width,
            args.height,
            rs.format.bgr8,
            args.fps,
        )

        self.started = False
        try:
            self.pipeline.start(config)
            self.started = True
            self.align = rs.align(rs.stream.color)
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.last_detection_at_s = None
            self.cached_rectangles = ()
            self.cached_scores = ()
        except Exception:
            self.close()
            raise

    def capture(self, include_rgb=False):
        frames = self.pipeline.wait_for_frames(self.args.frame_timeout_ms)
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("D435i returned an incomplete frame set")

        image = self.numpy.asanyarray(color_frame.get_data())
        detection_at_s = time.monotonic()
        if (
            self.last_detection_at_s is None
            or detection_at_s - self.last_detection_at_s
            >= 1.0 / self.args.detection_fps
        ):
            rectangles, scores = self.hog.detectMultiScale(
                image,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
            self.cached_rectangles = rectangles
            self.cached_scores = scores
            self.last_detection_at_s = detection_at_s
        else:
            rectangles = self.cached_rectangles
            scores = self.cached_scores
        accepted_scores = []
        distances = []
        target = None
        width = color_frame.get_width()
        height = color_frame.get_height()
        for rectangle, score_value in zip(rectangles, scores):
            score = float(score_value)
            if score < self.args.min_score or len(rectangle) < 4:
                continue
            x = int(rectangle[0])
            y = int(rectangle[1])
            box_width = int(rectangle[2])
            box_height = int(rectangle[3])
            center_x = min(max(x + box_width // 2, 0), width - 1)
            center_y = min(max(y + box_height // 2, 0), height - 1)
            distance = float(depth_frame.get_distance(center_x, center_y))
            if distance > 0:
                if (
                    self.args.max_distance_m is not None
                    and distance > self.args.max_distance_m
                ):
                    continue
                distances.append(distance)
                if target is None or distance < target[0]:
                    target = (distance, (center_x + 0.5) / width)
            accepted_scores.append(score)

        obstacle_distances = []
        # Avoid the lower image edge, which is commonly the floor on a
        # low-mounted D435i and otherwise blocks follow_person at all times.
        for row in range(6):
            sample_y = round(height * (0.18 + row * 0.08))
            for column in range(7):
                sample_x = round(width * (0.25 + column * 0.0833))
                distance = float(depth_frame.get_distance(sample_x, sample_y))
                if distance > 0:
                    obstacle_distances.append(distance)
        obstacle_distances.sort()
        nearest_obstacle_distance_m = None
        if obstacle_distances:
            index = max(0, round((len(obstacle_distances) - 1) * 0.1))
            nearest_obstacle_distance_m = obstacle_distances[index]

        result = {
            "observed_at_s": time.monotonic(),
            "person_count": len(accepted_scores),
            "nearest_person_distance_m": min(distances) if distances else None,
            "person_center_x": target[1] if target is not None else None,
            "confidence": (
                confidence(max(accepted_scores)) if accepted_scores else None
            ),
            "source": "realsense:%s" % (self.args.serial or "D435i"),
            "nearest_obstacle_distance_m": nearest_obstacle_distance_m,
        }
        if include_rgb:
            output_image = rotate_bgr_image(
                self.cv2, image, self.args.rgb_rotation_deg
            )
            success, encoded = self.cv2.imencode(
                ".jpg",
                output_image,
                [int(self.cv2.IMWRITE_JPEG_QUALITY), 80],
            )
            if not success:
                raise RuntimeError("failed to encode D435i RGB frame")
            result["rgb_jpeg_base64"] = base64.b64encode(encoded).decode("ascii")
        return result

    def close(self):
        if self.started:
            self.started = False
            self.pipeline.stop()


def main():
    detector = None
    try:
        detector = Detector(parse_args())
        send({"type": "ready"})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                command = request.get("command")
                if command == "capture":
                    send(
                        {
                            "type": "observation",
                            "result": detector.capture(
                                include_rgb=bool(request.get("include_rgb"))
                            ),
                        }
                    )
                elif command == "close":
                    detector.close()
                    send({"type": "closed"})
                    return 0
                else:
                    send({"type": "error", "message": "unknown worker command"})
            except Exception as exc:  # noqa: BLE001
                send({"type": "error", "message": f"D435i capture failed: {exc}"})
    except Exception as exc:  # noqa: BLE001
        send({"type": "error", "message": f"failed to open RealSense D435i: {exc}"})
        return 1
    finally:
        if detector is not None:
            try:
                detector.close()
            except Exception:  # noqa: BLE001, S110
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
