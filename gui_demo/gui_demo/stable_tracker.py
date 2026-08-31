"""Lightweight detector/Lucas-Kanade tracker fusion for GUI video frames."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence, TypeVar

import cv2
import numpy as np
from PIL import Image


MatchT = TypeVar("MatchT")
Box = tuple[int, int, int, int]


def box_iou(first: Box | None, second: Box | None) -> float:
    if first is None or second is None:
        return 0.0
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1)


def _blend_box(previous: Box | None, detected: Box | None, alpha: float) -> Box | None:
    if previous is None:
        return detected
    if detected is None:
        return previous
    return tuple(
        int(round((1.0 - alpha) * old + alpha * new))
        for old, new in zip(previous, detected)
    )  # type: ignore[return-value]


def _transform_box(box: Box | None, matrix: np.ndarray, width: int, height: int) -> Box | None:
    if box is None:
        return None
    corners = np.float32([[box[0], box[1]], [box[2], box[3]]]).reshape(-1, 1, 2)
    moved = cv2.transform(corners, matrix).reshape(-1, 2)
    x1, y1 = np.floor(moved.min(axis=0)).astype(int)
    x2, y2 = np.ceil(moved.max(axis=0)).astype(int)
    x1, x2 = int(np.clip(x1, 0, width - 1)), int(np.clip(x2, 1, width))
    y1, y2 = int(np.clip(y1, 0, height - 1)), int(np.clip(y2, 1, height))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else box


class _Track:
    def __init__(self, match: MatchT, points: np.ndarray, track_id: int) -> None:
        self.match = match
        self.points = points
        self.track_id = track_id
        self.missed_detections = 0


class LKBoxTracker:
    """Propagate boxes with pyramidal LK and softly correct them with detections.

    ``step`` receives detections only on detector frames. Passing ``None`` means
    no detector was run; passing an empty sequence means it ran but found none.
    The match objects must be dataclasses containing ``box``, ``face_box``,
    ``body_box`` and ``crop`` fields (the GUI's ``FrameMatch`` contract).
    """

    def __init__(
        self,
        crop_factory: Callable[[np.ndarray, Box], Image.Image] | None = None,
        detection_alpha: float = 0.42,
        max_missed_detections: int = 4,
        max_tracks: int | None = None,
    ) -> None:
        self.crop_factory = crop_factory or self._default_crop
        self.detection_alpha = detection_alpha
        self.max_missed_detections = max_missed_detections
        self.max_tracks = max_tracks
        self._previous_gray: np.ndarray | None = None
        self._tracks: list[_Track] = []
        self._next_id = 1

    @staticmethod
    def _default_crop(frame: np.ndarray, box: Box) -> Image.Image:
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((16, 16, 3), dtype=np.uint8)
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    def reset(self) -> None:
        self._previous_gray = None
        self._tracks = []
        self._next_id = 1

    @staticmethod
    def _seed_points(gray: np.ndarray, box: Box) -> np.ndarray:
        height, width = gray.shape
        x1, y1, x2, y2 = box
        x1, x2 = int(np.clip(x1, 0, width - 1)), int(np.clip(x2, 1, width))
        y1, y2 = int(np.clip(y1, 0, height - 1)), int(np.clip(y2, 1, height))
        mask = np.zeros_like(gray)
        inset_x, inset_y = max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)
        mask[y1 + inset_y:y2 - inset_y, x1 + inset_x:x2 - inset_x] = 255
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=100, qualityLevel=0.008, minDistance=5,
            blockSize=5, mask=mask,
        )
        if points is not None and len(points) >= 6:
            return points.astype(np.float32)
        xs = np.linspace(x1 + inset_x, x2 - inset_x, 5)
        ys = np.linspace(y1 + inset_y, y2 - inset_y, 4)
        return np.float32([(x, y) for y in ys for x in xs]).reshape(-1, 1, 2)

    def _propagate(self, gray: np.ndarray, frame: np.ndarray) -> None:
        if self._previous_gray is None:
            return
        height, width = gray.shape
        for track in self._tracks:
            if len(track.points) < 6:
                track.points = self._seed_points(self._previous_gray, track.match.box)
            forward, status, _ = cv2.calcOpticalFlowPyrLK(
                self._previous_gray, gray, track.points, None,
                winSize=(25, 25), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
            )
            if forward is None or status is None:
                track.points = np.empty((0, 1, 2), dtype=np.float32)
                continue
            backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, self._previous_gray, forward, None,
                winSize=(25, 25), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
            )
            if backward is None or reverse_status is None:
                track.points = np.empty((0, 1, 2), dtype=np.float32)
                continue
            error = np.linalg.norm(track.points.reshape(-1, 2) - backward.reshape(-1, 2), axis=1)
            valid = (status.ravel() == 1) & (reverse_status.ravel() == 1) & (error < 1.5)
            old_points = track.points.reshape(-1, 2)[valid]
            new_points = forward.reshape(-1, 2)[valid]
            if len(new_points) < 4:
                track.points = np.empty((0, 1, 2), dtype=np.float32)
                continue
            matrix, _ = cv2.estimateAffinePartial2D(
                old_points, new_points, method=cv2.RANSAC,
                ransacReprojThreshold=2.5, maxIters=100, confidence=0.98,
            )
            if matrix is None:
                delta = np.median(new_points - old_points, axis=0)
                matrix = np.float32([[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]]])
            scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
            if not 0.88 <= scale <= 1.14:
                delta = np.median(new_points - old_points, axis=0)
                matrix = np.float32([[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]]])
            main_box = _transform_box(track.match.box, matrix, width, height)
            face_box = _transform_box(getattr(track.match, "face_box", None), matrix, width, height)
            body_box = _transform_box(getattr(track.match, "body_box", None), matrix, width, height)
            if main_box is not None:
                track.match = replace(
                    track.match, box=main_box, face_box=face_box, body_box=body_box,
                    crop=self.crop_factory(frame, main_box),
                )
            track.points = new_points.reshape(-1, 1, 2).astype(np.float32)

    def _merge_detections(self, gray: np.ndarray, frame: np.ndarray, detections: Sequence[MatchT]) -> None:
        if self.max_tracks == 1 and self._tracks and len(detections) == 1:
            # Single-target mode deliberately preserves one identity. Fast
            # motion can eliminate IoU between sparse detector frames, so a
            # sole high-quality detection is an unconditional correction.
            detection = detections[0]
            track = self._tracks[0]
            box = _blend_box(track.match.box, detection.box, self.detection_alpha)
            face_box = _blend_box(
                getattr(track.match, "face_box", None), getattr(detection, "face_box", None),
                self.detection_alpha,
            )
            body_box = _blend_box(
                getattr(track.match, "body_box", None), getattr(detection, "body_box", None),
                self.detection_alpha,
            )
            track.match = replace(
                detection, box=box, face_box=face_box, body_box=body_box,
                crop=self.crop_factory(frame, box),
            )
            track.points = self._seed_points(gray, box)
            track.missed_detections = 0
            self._tracks = [track]
            return
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                score = max(
                    box_iou(getattr(track.match, "body_box", None), getattr(detection, "body_box", None)),
                    box_iou(getattr(track.match, "face_box", None), getattr(detection, "face_box", None)),
                    box_iou(track.match.box, detection.box),
                )
                if score >= 0.12:
                    candidates.append((score, track_index, detection_index))
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            track, detection = self._tracks[track_index], detections[detection_index]
            box = _blend_box(track.match.box, detection.box, self.detection_alpha)
            face_box = _blend_box(
                getattr(track.match, "face_box", None), getattr(detection, "face_box", None),
                self.detection_alpha,
            )
            body_box = _blend_box(
                getattr(track.match, "body_box", None), getattr(detection, "body_box", None),
                self.detection_alpha,
            )
            track.match = replace(
                detection, box=box, face_box=face_box, body_box=body_box,
                crop=self.crop_factory(frame, box),
                score=0.65 * float(getattr(track.match, "score", detection.score)) + 0.35 * detection.score,
            )
            track.points = self._seed_points(gray, box)
            track.missed_detections = 0
            used_tracks.add(track_index)
            used_detections.add(detection_index)
        survivors: list[_Track] = []
        for index, track in enumerate(self._tracks):
            if index not in used_tracks:
                track.missed_detections += 1
            if track.missed_detections <= self.max_missed_detections and len(track.points) >= 4:
                survivors.append(track)
        self._tracks = survivors
        for index, detection in enumerate(detections):
            if index in used_detections:
                continue
            if self.max_tracks is not None and len(self._tracks) >= self.max_tracks:
                break
            track = _Track(detection, self._seed_points(gray, detection.box), self._next_id)
            self._next_id += 1
            self._tracks.append(track)

    def step(self, frame: np.ndarray, detections: Sequence[MatchT] | None = None) -> list[MatchT]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._propagate(gray, frame)
        if detections is not None:
            self._merge_detections(gray, frame, detections)
        self._previous_gray = gray
        return [track.match for track in self._tracks]
