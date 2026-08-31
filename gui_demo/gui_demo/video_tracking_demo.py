#!/usr/bin/env python3
"""Register a cat from video and export detector + LK stabilized boxes."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from run_gui import CatFaceService, REGISTRATION_FACE_CONFIDENCE
from stable_tracker import LKBoxTracker


@dataclass
class Candidate:
    frame_index: int
    frame: np.ndarray
    confidence: float
    sharpness: float
    face_area: float

    @property
    def quality(self) -> float:
        # Confidence rejects detector noise; sharpness breaks ties between
        # nearby poses without allowing a tiny, sharp face to dominate.
        return self.confidence * math.log1p(self.sharpness) * math.sqrt(self.face_area)


def _video_info(path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return fps, count, width, height


def collect_candidates(
    service: CatFaceService, video_path: Path, start_seconds: float,
    end_seconds: float | None = None, batch_size: int = 24,
) -> tuple[list[Candidate], float]:
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    start_frame = max(0, round(start_seconds * fps))
    end_frame = round(end_seconds * fps) if end_seconds is not None else None
    batch: list[np.ndarray] = []
    indices: list[int] = []
    candidates: list[Candidate] = []
    frame_index = 0

    def infer_batch() -> None:
        if not batch:
            return
        face_groups, _, _, _, _ = service._detect_modalities(batch)
        for index, frame, faces in zip(indices, batch, face_groups):
            if not faces:
                continue
            face = max(faces, key=lambda item: item.confidence)
            gray_crop = cv2.cvtColor(np.asarray(face.crop), cv2.COLOR_RGB2GRAY)
            sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
            candidates.append(Candidate(
                frame_index=index, frame=frame.copy(), confidence=face.confidence,
                sharpness=sharpness, face_area=face.face_area,
            ))
        batch.clear()
        indices.clear()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if end_frame is not None and frame_index > end_frame:
            break
        if frame_index >= start_frame:
            batch.append(frame)
            indices.append(frame_index)
            if len(batch) >= batch_size:
                infer_batch()
        frame_index += 1
    infer_batch()
    capture.release()
    return candidates, fps


def select_registration_frames(
    candidates: list[Candidate], fps: float, count: int = 5,
) -> list[Candidate]:
    # Registration requires temporal evidence, not a single detector hit.
    # This removes isolated false positives on toys/background objects after
    # the cat has left the scene.
    support_radius = max(2, round(0.25 * fps))
    stable_candidates = [
        candidate for candidate in candidates
        if candidate.confidence >= REGISTRATION_FACE_CONFIDENCE
        if sum(
            abs(candidate.frame_index - neighbor.frame_index) <= support_radius
            for neighbor in candidates
        ) >= 3
    ]
    if len(stable_candidates) < count:
        raise ValueError(
            f"2秒后只有 {len(stable_candidates)} 帧通过连续猫脸校验，无法选满 {count} 张注册图"
        )
    # Prefer distinct moments/poses. Relax spacing only if the clip contains
    # fewer separated detections than requested.
    for spacing_seconds in (0.35, 0.22, 0.10, 0.0):
        minimum_gap = round(spacing_seconds * fps)
        selected: list[Candidate] = []
        for candidate in sorted(stable_candidates, key=lambda item: item.quality, reverse=True):
            if all(abs(candidate.frame_index - item.frame_index) >= minimum_gap for item in selected):
                selected.append(candidate)
                if len(selected) == count:
                    return sorted(selected, key=lambda item: item.frame_index)
    raise ValueError("无法选出足够的注册帧")


def register_from_video(
    service: CatFaceService, video_path: Path, identity: str,
    start_seconds: float, end_seconds: float | None = None,
) -> list[dict[str, float | int]]:
    candidates, fps = collect_candidates(
        service, video_path, start_seconds, end_seconds=end_seconds,
    )
    selected = select_registration_frames(candidates, fps)
    with tempfile.TemporaryDirectory(prefix="cat-video-register-") as temporary:
        paths: list[Path] = []
        for order, candidate in enumerate(selected, start=1):
            # Lossless storage matters here: this clip's face detections sit
            # close to the confidence threshold and JPEG ringing can make a
            # frame pass selection but fail the registry's mandatory recheck.
            path = Path(temporary) / f"{order:02d}_frame_{candidate.frame_index:06d}.png"
            if not cv2.imwrite(str(path), candidate.frame, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
                raise OSError(f"注册帧写入失败：{path}")
            paths.append(path)
        service.register(identity, paths)
    return [
        {
            "frame": item.frame_index,
            "seconds": round(item.frame_index / fps, 3),
            "confidence": round(item.confidence, 4),
            "sharpness": round(item.sharpness, 2),
            "face_area": round(item.face_area, 6),
        }
        for item in selected
    ]


def export_tracked_video(
    service: CatFaceService, input_path: Path, output_path: Path,
    detect_interval: int, target_identity: str = "",
) -> dict[str, int | float | str]:
    fps, frame_count, width, height = _video_info(input_path)
    capture = cv2.VideoCapture(str(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"无法创建输出视频：{output_path}")
    tracker = LKBoxTracker(
        detection_alpha=0.38, max_missed_detections=18, max_tracks=1,
    )
    detected_frames = 0
    tracked_frames = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = None
            if frame_index % detect_interval == 0:
                detections, _ = service.identify_frame(frame, adaptive=False)
                if target_identity:
                    frame_area = width * height
                    candidates = [
                        match for match in detections
                        if (
                            match.body_box is not None
                            and (match.body_box[2] - match.body_box[0])
                            * (match.body_box[3] - match.body_box[1]) / frame_area >= 0.015
                        ) or (
                            match.face_box is not None
                            and (match.face_box[2] - match.face_box[0])
                            * (match.face_box[3] - match.face_box[1]) / frame_area >= 0.018
                        )
                    ]
                    if candidates:
                        best = max(
                            candidates,
                            key=lambda match: (
                                (match.body_box[2] - match.body_box[0])
                                * (match.body_box[3] - match.body_box[1])
                                if match.body_box is not None else 0,
                                match.confidence,
                            ),
                        )
                        main_box = best.body_box or best.face_box
                        detections = [replace(
                            best, identity=target_identity, passed=True,
                            box=main_box, body_box=main_box, face_box=None,
                        )]
                    else:
                        detections = []
                detected_frames += int(bool(detections))
            matches = tracker.step(frame, detections)
            tracked_frames += int(bool(matches))
            annotated = service.annotate_frame(frame, matches)
            writer.write(cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR))
            frame_index += 1
            if frame_index % 60 == 0:
                print(f"处理进度：{frame_index}/{frame_count} 帧")
    finally:
        capture.release()
        writer.release()
    return {
        "frames": frame_index,
        "fps": fps,
        "detector_hit_frames": detected_frames,
        "frames_with_stable_box": tracked_frames,
        "output": str(output_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="猫脸注册 + YOLO/LK 稳定目标跟踪视频 Demo")
    parser.add_argument("video", type=Path, help="输入视频")
    parser.add_argument("--register-id", default="", help="从视频自动注册的新身份，例如：跑得快")
    parser.add_argument("--register-after", type=float, default=2.0, help="从第几秒开始挑注册帧")
    parser.add_argument("--register-before", type=float, default=None, help="到第几秒停止挑注册帧")
    parser.add_argument("--detect-interval", type=int, default=3, help="每隔多少帧运行一次 YOLO")
    parser.add_argument("--output", type=Path, default=Path("tracked_output.mp4"), help="输出 MP4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.detect_interval < 1:
        raise ValueError("--detect-interval 必须大于等于 1")
    service = CatFaceService()
    try:
        print("正在加载检测和识别模型…")
        service.load()
        report: dict[str, object] = {}
        if args.register_id.strip():
            print(f"正在从 {args.register_after:.1f} 秒后选择清晰帧，注册“{args.register_id.strip()}”…")
            report["registration"] = register_from_video(
                service, args.video, args.register_id.strip(), args.register_after,
                args.register_before,
            )
        print("正在导出 YOLO + LK 稳定跟踪视频…")
        report["tracking"] = export_tracked_video(
            service, args.video, args.output, args.detect_interval,
            target_identity=args.register_id.strip(),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
