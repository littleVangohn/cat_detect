#!/usr/bin/env python3
r"""Local closed-set cat face identification desktop GUI.

Two pages: still-photo and live-camera multi-cat recognition.
Launch with (from the repository root):
    python gui_demo/run_gui.py
All file paths are resolved relative to this file, so the folder is
portable across Windows / Linux as long as ../data/gui_registry exists.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import ctypes
import hashlib
import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk
from timm.utils import reparameterize_model
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
from mobileone_embedder import StudentModel  # noqa: E402
from stable_tracker import LKBoxTracker  # noqa: E402

DETECTOR_PATH = ROOT / "models" / "cat_body_face_yolo11n_v1.pt"
CHECKPOINT_PATH = ROOT / "models" / "face_mobileone_s1.pt"
BODY_CHECKPOINT_PATH = ROOT / "models" / "body_mobileone_s1.pt"
REGISTRY_ROOT = ROOT.parent / "data" / "gui_registry"
REGISTRY_IMAGES = REGISTRY_ROOT / "images"
REGISTRY_VECTORS = REGISTRY_ROOT / "vectors"
REGISTRY_MANIFEST = REGISTRY_ROOT / "registry.json"
UI_FONT_PATH = ROOT / "assets" / "ui_font.ttf"
UI_FONT_FAMILY = "猫啃什锦黑-轻量版"
HEADER_ART_PATH = ROOT / "assets" / "header.png"

BG = "#D5EAE3"
TEXT = "#775C55"
FIELD = "#F8F4E9"
PINK = "#FDD3D5"
BODY_BOX = "#2F9E44"  # 身框描边色，与脸框粉色区分
DIMENSION = 384
CONFIDENCE = 0.10
REGISTRATION_FACE_CONFIDENCE = 0.15
BODY_CONFIDENCE = 0.10
BODY_CLASS_ID = 0
FACE_CLASS_ID = 1
FACE_WEIGHT = 0.62
BODY_WEIGHT = 0.38
BODY_MARGIN = 0.15
CAMERA_WIDTH = 1280  # 摄像头采集分辨率（不支持的设备会自动回退）
CAMERA_HEIGHT = 720
CAMERA_PREVIEW_WIDTH = 960  # 仅限制界面预览；检测仍使用原始采集帧
CAMERA_PREVIEW_HEIGHT = 540
PHOTO_PREVIEW_WIDTH = 1600  # 仅限制带框展示图；身份特征仍取自原始照片
PHOTO_PREVIEW_HEIGHT = 1200
NMS_IOU = 0.50  # 更早抑制同一张脸的重复框，同时保留并排猫脸
IMGSZ = 640
MARGIN = 0.10
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 匹配侧动态阈值（与 demo_hisi/board_src/include/types.hpp MatcherConfig 一致）：
# 按脸框面积占画面比例分三档，小脸特征噪声大、门槛更严；第一名还需比第二名
# 高 MATCH_MARGIN 才算高置信；界面仍按用户要求直接显示最相近身份。
FIXED_MATCH_THRESHOLD = 0.35
MATCH_MARGIN = 0.03
LARGE_FACE_AREA = 0.35
MEDIUM_FACE_AREA = 0.10
LARGE_FACE_THRESHOLD = 0.40
MEDIUM_FACE_THRESHOLD = 0.45
SMALL_FACE_THRESHOLD = 0.50
DEFAULT_IMAGES_PER_IDENTITY = 5
LEGACY_IMAGES_PER_IDENTITY = 6


def ui_font(size: int, bold: bool = False) -> tuple[str, int] | tuple[str, int, str]:
    # The bundled Lite font only contains Regular. Requesting bold makes Tk
    # synthesize heavier pixels, which looks fuzzy on high-DPI displays.
    return (UI_FONT_FAMILY, size)


def enable_high_dpi() -> None:
    """Prevent Windows from bitmap-scaling the whole Tk window."""
    if os.name != "nt":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass


def register_ui_font() -> None:
    """Expose the bundled font to Tk for this process without installing it system-wide.

    Windows: AddFontResourceExW registers the font for this process only.
    Linux/macOS: copy the font into the user fontconfig directory before Tk
    starts, so the Tk window and PIL annotation both resolve the family.
    """
    if not UI_FONT_PATH.is_file():
        raise FileNotFoundError(f"缺少界面字体：{UI_FONT_PATH}")
    if os.name == "nt":
        added = ctypes.windll.gdi32.AddFontResourceExW(str(UI_FONT_PATH), 0x10, 0)
        if added == 0:
            raise RuntimeError(f"无法加载界面字体：{UI_FONT_PATH}")
    else:
        try:
            font_dir = Path.home() / ".local" / "share" / "fonts"
            font_dir.mkdir(parents=True, exist_ok=True)
            target = font_dir / UI_FONT_PATH.name
            if not target.is_file() or target.stat().st_size != UI_FONT_PATH.stat().st_size:
                shutil.copy2(UI_FONT_PATH, target)
            subprocess.run(
                ["fc-cache", "-f", str(font_dir)],
                check=False, capture_output=True, timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            pass  # 装不上就退化为系统默认字体，不影响运行


@dataclass
class GalleryItem:
    identity: str
    path: Path
    feature: np.ndarray
    source: str
    body_feature: np.ndarray | None = None


@dataclass
class DetectedFace:
    box: tuple[int, int, int, int]
    crop: Image.Image
    confidence: float
    face_area: float


@dataclass
class DetectedBody:
    box: tuple[int, int, int, int]
    crop: Image.Image
    confidence: float
    body_area: float


@dataclass
class Timing:
    """一次识别（图片或摄像头一帧）的分阶段耗时（毫秒）。"""

    total_ms: float
    decode_ms: float = 0.0
    detect_ms: float = 0.0
    feature_ms: float = 0.0
    match_ms: float = 0.0
    association_ms: float = 0.0
    face_feature_ms: float = 0.0
    body_feature_ms: float = 0.0
    fusion_ms: float = 0.0


@dataclass
class MatchOutcome:
    """均值模板 + 动态阈值的匹配结果。"""

    identity: str
    score: float
    second_identity: str
    second_score: float
    threshold: float
    face_area: float
    passed: bool
    match_path: Path | None
    decision_margin: float = 0.0


@dataclass
class FrameMatch:
    """摄像头单帧中一只猫的识别结果。"""

    box: tuple[int, int, int, int]
    crop: Image.Image
    confidence: float
    identity: str
    score: float
    match_path: Path | None
    second_identity: str = ""
    second_score: float = 0.0
    threshold: float = 0.0
    passed: bool = True
    decision_margin: float = 0.0
    # 双 ROI 可视化：box 仍为主框（有身框取身框），face_box/body_box
    # 供界面同时绘制脸框与身框；两者均可能为 None（单分支检出）。
    face_box: tuple[int, int, int, int] | None = None
    body_box: tuple[int, int, int, int] | None = None


def timing_summary(timing: Timing) -> str:
    """Compact stage breakdown shared by still-image and camera pages."""
    unified_detection = timing.detect_ms
    dual_features = timing.face_feature_ms + timing.body_feature_ms
    return (
        f"总耗时 {timing.total_ms:.0f} ms（统一检测 {unified_detection:.0f}；"
        f"双特征 {dual_features:.0f}：脸 {timing.face_feature_ms:.0f}/"
        f"身体 {timing.body_feature_ms:.0f}；关联 {timing.association_ms:.1f}；"
        f"融合 {timing.fusion_ms:.1f}）"
    )


def rounded_rectangle(canvas: tk.Canvas, x1: int, y1: int, x2: int, y2: int, radius: int, **kwargs):
    radius = min(radius, max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
    points = [
        x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
        x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
        x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class BackgroundWorker:
    """常驻单线程任务执行器。

    GPU 推理有个特性：每个新线程的首次 CUDA 前向要付出 ~750 ms 的一次性
    开销（内核/引擎装载），之后同线程再调用只需十几毫秒。如果每次点击都
    新建线程跑推理，这笔开销就每次重付——界面表现为单张图片"检测"要
    ~0.8 s。把所有后台任务（模型加载、图片识别、身份注册）放在同一常驻
    线程里顺序执行，热身只发生一次。
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop, name="catface-worker", daemon=True,
        )
        self._thread.start()

    def submit(self, task, on_success, on_error) -> None:
        """提交任务；task 在线程中执行，on_success/on_error 也在该线程被调用，
        由调用方负责把 Tk 界面回调通过 root.after 转回主线程。"""
        self._queue.put((task, on_success, on_error))

    def _loop(self) -> None:
        while True:
            task, on_success, on_error = self._queue.get()
            try:
                value = task()
            except Exception as error:
                on_error(error)
            else:
                on_success(value)


class RoundedButton(tk.Canvas):
    """A small canvas button because Tk's native Button has square corners."""

    def __init__(self, parent, text: str, command, **kwargs):
        super().__init__(parent, height=46, bg=BG, highlightthickness=0, cursor="hand2", **kwargs)
        self.text, self.command, self.enabled = text, command, True
        self.active = False
        self.bind("<Configure>", lambda _event: self._draw())
        self.bind("<Button-1>", self._click)
        self.bind("<Enter>", lambda _event: self._draw(hover=True))
        self.bind("<Leave>", lambda _event: self._draw())
        self._draw()

    def set_active(self, active: bool) -> None:
        self.active = active
        self._draw()

    def set_text(self, text: str) -> None:
        self.text = text
        self._draw()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        self._draw()

    def _click(self, _event) -> None:
        if self.enabled:
            self.command()

    def _draw(self, hover: bool = False) -> None:
        self.delete("all")
        width, height = max(120, self.winfo_width()), max(42, self.winfo_height())
        if self.active:
            fill = "#A9D3C0"  # 当前页高亮
        elif hover and self.enabled:
            fill = FIELD
        elif not self.enabled:
            fill = "#E8DCDD"
        else:
            fill = PINK
        rounded_rectangle(self, 3, 5, width - 2, height - 1, 20, fill="#C5D9D2", outline="")
        rounded_rectangle(self, 2, 2, width - 2, height - 4, 20, fill=fill, outline=FIELD, width=2)
        rounded_rectangle(self, 5, 5, width - 5, height - 7, 17, fill="", outline="#FFFDF7", width=1)
        self.create_text(width // 2, height // 2 - 1, text=self.text, fill=TEXT, font=ui_font(12, bold=True))


class RoundedImageCard(tk.Canvas):
    """Fixed preview surface that letterboxes images instead of cropping them."""

    def __init__(
        self, parent, width: int = 300, height: int = 292,
        min_w: int = 240, min_h: int = 230,
        resample: Image.Resampling = Image.Resampling.LANCZOS,
        allow_upscale: bool = True, **kwargs,
    ):
        super().__init__(parent, width=width, height=height, bg=BG, highlightthickness=0, **kwargs)
        self.min_w, self.min_h = min_w, min_h
        self.resample = resample
        self.allow_upscale = allow_upscale
        self.source_image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.bind("<Configure>", lambda _event: self._draw())
        self._draw()

    def set_image(self, image: Image.Image) -> None:
        self.source_image = image if image.mode == "RGB" else image.convert("RGB")
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        width, height = max(self.min_w, self.winfo_width()), max(self.min_h, self.winfo_height())
        rounded_rectangle(self, 5, 7, width - 3, height - 2, 25, fill="#BED8D0", outline="")
        rounded_rectangle(self, 3, 3, width - 5, height - 6, 25, fill=FIELD, outline=PINK, width=2)
        rounded_rectangle(self, 7, 7, width - 9, height - 10, 21, fill="", outline="#FFFFFF", width=1)
        if self.source_image is None:
            self.create_text(width // 2, height // 2, text="暂无图片", fill=TEXT, font=ui_font(12))
            return
        inner_w, inner_h = max(40, width - 28), max(40, height - 28)
        target = (inner_w, inner_h)
        if not self.allow_upscale:
            target = (min(inner_w, self.source_image.width), min(inner_h, self.source_image.height))
        shown = ImageOps.contain(self.source_image, target, self.resample)
        self.photo = ImageTk.PhotoImage(shown)
        self.create_image(width // 2, height // 2, image=self.photo)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)


def expanded_box(box: Iterable[float], width: int, height: int) -> tuple[int, int, int, int]:
    return expanded_box_with_margin(box, width, height, MARGIN)


def expanded_box_with_margin(
    box: Iterable[float], width: int, height: int, margin: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    left, top = max(0, int(math.floor(x1 - dx))), max(0, int(math.floor(y1 - dy)))
    right, bottom = min(width, int(math.ceil(x2 + dx))), min(height, int(math.ceil(y2 + dy)))
    if right <= left or bottom <= top:
        raise ValueError("检测框扩展后无有效区域")
    return left, top, right, bottom


class CatFaceService:
    def __init__(self, *, deploy_models: bool = True) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_arg = str(self.device)
        self.deploy_models = deploy_models
        self.detector: YOLO | None = None
        self.model: torch.nn.Module | None = None
        self.body_model: torch.nn.Module | None = None
        self.gallery: list[GalleryItem] = []
        self.default_gallery: list[GalleryItem] = []
        self.default_ids: list[str] = []
        self._gallery_lock = threading.Lock()  # 摄像头线程与注册操作共享图库
        self._inference_lock = threading.RLock()
        self._template_names: list[str] = []
        self._template_matrix = np.zeros((0, DIMENSION), dtype=np.float32)
        self._body_template_matrix = np.zeros((0, DIMENSION), dtype=np.float32)
        self._identity_features: dict[str, np.ndarray] = {}
        self._identity_body_features: dict[str, np.ndarray] = {}
        self._identity_paths: dict[str, list[Path]] = {}
        self._annotation_fonts: dict[int, ImageFont.ImageFont] = {}
        self._norm_mean: torch.Tensor | None = None
        self._norm_std: torch.Tensor | None = None

    def load(self) -> None:
        if not BODY_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError("Missing body identity model")
        if not DETECTOR_PATH.is_file() or not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError("缺少 YOLO11n 或 MobileOne-S1 权重")
        self.detector = YOLO(str(DETECTOR_PATH))
        model = StudentModel("mobileone_s1", pretrained=False)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        if self.deploy_models:
            model = reparameterize_model(model, inplace=False)
        self.model = model.to(self.device).eval()
        body_model = StudentModel("mobileone_s1", pretrained=False)
        body_checkpoint = torch.load(BODY_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        body_model.load_state_dict(body_checkpoint["state_dict"])
        if self.deploy_models:
            body_model = reparameterize_model(body_model, inplace=False)
        self.body_model = body_model.to(self.device).eval()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self._norm_mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        self._norm_std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        self.default_gallery = self._build_default_gallery()
        self.default_ids = sorted({item.identity for item in self.default_gallery})
        if not self.default_ids:
            raise RuntimeError("gui_registry/images 中没有可用的默认身份")
        with self._gallery_lock:
            self.gallery = list(self.default_gallery)
            self._load_user_registry()
            self._populate_body_gallery_features()
            self._rebuild_templates()
        self._warmup()

    def close(self) -> None:
        """Release service-owned resources (models are reclaimed by the process)."""

    def _warmup(self) -> None:
        """Warm the exact batch-1 runtime path used by photo and camera inference."""
        blank = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
        crop = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        with self._inference_lock:
            for _ in range(2):
                self._detect_modalities([blank])
                self._embed_crops([crop], flip_tta=False)
                self._embed_crops([crop], flip_tta=False, model=self.body_model)

    def _build_default_gallery(self) -> list[GalleryItem]:
        # 与板端五猫注册库同源：取 gui_registry/images 中未注册的
        # 身份目录（现为 cat_224149 / cat_226675），注册身份由 registry.json
        # 单独加载，避免重复进库。新库每个身份使用 5 张注册图。
        manifest = self._manifest()
        safe_to_name = {entry["safe_id"]: entry["identity"] for entry in manifest["identities"]}
        registered = set(safe_to_name.values())
        candidates: list[tuple[str, Path]] = []
        for directory in sorted(REGISTRY_IMAGES.iterdir()):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            identity = safe_to_name.get(directory.name, directory.name)
            if identity in registered:
                continue  # 已注册身份由 _load_user_registry 加载
            paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_TYPES)
            candidates.extend((identity, path) for path in paths[:DEFAULT_IMAGES_PER_IDENTITY])
        output: list[GalleryItem] = []
        if candidates:
            extracted, body_extracted, _ = self._extract_primary_modalities(
                [path for _, path in candidates]
            )
            for (identity, path), feature, body_feature in zip(candidates, extracted, body_extracted):
                if feature is not None:
                    output.append(GalleryItem(
                        identity, path, feature, "default", body_feature=body_feature,
                    ))
        return output

    def _faces_from_result(self, frame_bgr: np.ndarray, result) -> list[DetectedFace]:
        """Convert one YOLO result to stable left-to-right face crops."""
        boxes = result.boxes
        if len(boxes) == 0:
            return []
        height, width = frame_bgr.shape[:2]
        faces: list[DetectedFace] = []
        raw_boxes = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
        for raw_box, confidence, class_id in zip(raw_boxes, confidences, classes):
            if int(class_id) != FACE_CLASS_ID or float(confidence) < CONFIDENCE:
                continue
            try:
                left, top, right, bottom = expanded_box(raw_box, width, height)
            except ValueError:
                continue
            resized = cv2.resize(
                frame_bgr[top:bottom, left:right], (224, 224), interpolation=cv2.INTER_CUBIC,
            )
            crop = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            area = (raw_box[2] - raw_box[0]) * (raw_box[3] - raw_box[1]) / (width * height)
            faces.append(DetectedFace(
                box=tuple(int(round(value)) for value in raw_box),
                crop=crop,
                confidence=float(confidence),
                face_area=float(area),
            ))
        return sorted(faces, key=lambda face: ((face.box[0] + face.box[2]) / 2, face.box[1]))

    def _bodies_from_result(self, frame_bgr: np.ndarray, result) -> list[DetectedBody]:
        boxes = result.boxes
        if len(boxes) == 0:
            return []
        height, width = frame_bgr.shape[:2]
        bodies: list[DetectedBody] = []
        raw_boxes = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
        for raw_box, confidence, class_id in zip(raw_boxes, confidences, classes):
            if int(class_id) != BODY_CLASS_ID or float(confidence) < BODY_CONFIDENCE:
                continue
            try:
                left, top, right, bottom = expanded_box_with_margin(
                    raw_box, width, height, BODY_MARGIN
                )
            except ValueError:
                continue
            resized = cv2.resize(
                frame_bgr[top:bottom, left:right], (224, 224), interpolation=cv2.INTER_CUBIC,
            )
            crop = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            area = (raw_box[2] - raw_box[0]) * (raw_box[3] - raw_box[1]) / (width * height)
            bodies.append(DetectedBody(
                box=tuple(int(round(value)) for value in raw_box),
                crop=crop, confidence=float(confidence), body_area=float(area),
            ))
        return sorted(bodies, key=lambda body: ((body.box[0] + body.box[2]) / 2, body.box[1]))

    def _detect_modalities(
        self, frames_bgr: list[np.ndarray],
    ) -> tuple[list[list[DetectedFace]], list[list[DetectedBody]], float, float, float]:
        """Run one dual-class YOLO pass, then split cat-face and cat-body ROIs."""
        if self.detector is None:
            raise RuntimeError("Detector is not loaded")
        started = time.perf_counter()
        results = self.detector.predict(
            source=frames_bgr, imgsz=IMGSZ,
            conf=min(CONFIDENCE, BODY_CONFIDENCE), iou=NMS_IOU,
            classes=[BODY_CLASS_ID, FACE_CLASS_ID],
            device=self.device_arg, verbose=False, save=False, rect=False,
        )
        detected_ms = (time.perf_counter() - started) * 1000.0
        face_groups = [
            self._faces_from_result(frame, result)
            for frame, result in zip(frames_bgr, results)
        ]
        body_groups = [
            self._bodies_from_result(frame, result)
            for frame, result in zip(frames_bgr, results)
        ]
        wall_ms = (time.perf_counter() - started) * 1000.0
        # Keep the existing timing schema. Both class timings refer to the same
        # physical detector pass and must not be added together.
        return face_groups, body_groups, detected_ms, detected_ms, wall_ms

    def _embed_crops(
        self, crops: list[Image.Image], flip_tta: bool = False,
        model: torch.nn.Module | None = None,
    ) -> np.ndarray:
        """Embed crops in one GPU batch; flip TTA is reserved for gallery construction."""
        selected_model = model or self.model
        if selected_model is None or self._norm_mean is None or self._norm_std is None:
            raise RuntimeError("模型尚未加载")
        if not crops:
            return np.zeros((0, DIMENSION), dtype=np.float32)
        array = np.stack([
            np.asarray(crop, dtype=np.uint8).transpose(2, 0, 1)
            for crop in crops
        ])
        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(self.device, non_blocking=True)
        tensor = tensor.float().div_(255.0).sub_(self._norm_mean).div_(self._norm_std)
        batch_size = len(crops)
        with torch.inference_mode():
            model_input = torch.cat((tensor, torch.flip(tensor, dims=(3,))), dim=0) if flip_tta else tensor
            features = F.normalize(selected_model(model_input).float(), dim=1)
            if flip_tta:
                features = F.normalize(features[:batch_size] + features[batch_size:], dim=1)
        return features.cpu().numpy().astype(np.float32, copy=False)

    def _extract_primary_modalities(
        self, paths: list[Path],
    ) -> tuple[list[np.ndarray | None], list[np.ndarray | None], list[float]]:
        """Extract face/body enrollment features after one unified detection batch."""
        frames = [load_rgb(path)[:, :, ::-1].copy() for path in paths]
        face_groups, body_groups, _, _, _ = self._detect_modalities(frames)
        primary_faces = [max(faces, key=lambda face: face.confidence) if faces else None for faces in face_groups]
        primary_bodies = [max(bodies, key=lambda body: body.body_area) if bodies else None for bodies in body_groups]
        face_values = iter(self._embed_crops([face.crop for face in primary_faces if face is not None]))
        body_values = iter(self._embed_crops(
            [body.crop for body in primary_bodies if body is not None],
            flip_tta=True, model=self.body_model,
        ))
        faces = [next(face_values) if face is not None else None for face in primary_faces]
        bodies = [next(body_values) if body is not None else None for body in primary_bodies]
        confidences = [face.confidence if face is not None else 0.0 for face in primary_faces]
        return faces, bodies, confidences

    def _extract_primary_body_features(self, paths: list[Path]) -> list[np.ndarray | None]:
        frames = [load_rgb(path)[:, :, ::-1].copy() for path in paths]
        _, body_groups, _, _, _ = self._detect_modalities(frames)
        primary = [max(bodies, key=lambda body: body.body_area) if bodies else None for bodies in body_groups]
        valid = [body.crop for body in primary if body is not None]
        embedded = iter(self._embed_crops(valid, flip_tta=True, model=self.body_model))
        return [next(embedded) if body is not None else None for body in primary]

    def _populate_body_gallery_features(self) -> None:
        missing = [item for item in self.gallery if item.body_feature is None]
        if not missing:
            return
        features = self._extract_primary_body_features([item.path for item in missing])
        for item, feature in zip(missing, features):
            item.body_feature = feature

    def _manifest(self) -> dict:
        if not REGISTRY_MANIFEST.exists():
            return {"version": 1, "identities": []}
        try:
            value = json.loads(REGISTRY_MANIFEST.read_text(encoding="utf-8"))
            return value if isinstance(value.get("identities"), list) else {"version": 1, "identities": []}
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "identities": []}

    def _write_manifest(self, value: dict) -> None:
        REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = REGISTRY_MANIFEST.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, REGISTRY_MANIFEST)

    @staticmethod
    def _safe_id(identity: str) -> str:
        cleaned = re.sub(r"[^\w-]+", "_", identity, flags=re.UNICODE).strip("_")
        if not cleaned:
            raise ValueError("身份 ID 只能包含中文、字母、数字、下划线或连字符")
        # A stable suffix prevents two different display IDs from sharing one
        # on-disk directory after punctuation normalization.
        return f"{cleaned[:60]}_{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:10]}"

    def _load_user_registry(self) -> None:
        manifest = self._manifest()
        loaded = []
        for entry in manifest["identities"]:
            try:
                identity = str(entry["identity"])
                vector_path = REGISTRY_ROOT / entry["vector_file"]
                paths = [REGISTRY_ROOT / item for item in entry["image_files"]]
                vectors = np.load(vector_path).astype(np.float32)
                if len(paths) not in {DEFAULT_IMAGES_PER_IDENTITY, LEGACY_IMAGES_PER_IDENTITY}:
                    continue
                if vectors.shape != (len(paths), DIMENSION) or not all(path.is_file() for path in paths):
                    continue
                ordered = sorted(zip(paths, vectors), key=lambda item: str(item[0]).lower())
                paths = [item[0] for item in ordered[:DEFAULT_IMAGES_PER_IDENTITY]]
                vectors = np.stack([item[1] for item in ordered[:DEFAULT_IMAGES_PER_IDENTITY]])
                loaded.extend(GalleryItem(identity, path, vectors[index], "user") for index, path in enumerate(paths))
            except (KeyError, OSError, ValueError):
                continue
        self.gallery.extend(loaded)

    def register(self, identity: str, paths: list[Path]) -> None:
        identity = identity.strip()
        if not identity:
            raise ValueError("请输入身份 ID")
        if len(paths) != DEFAULT_IMAGES_PER_IDENTITY:
            raise ValueError("每个身份必须选择恰好5张图片")
        resolved = [path.resolve() for path in paths]
        if len(set(resolved)) != DEFAULT_IMAGES_PER_IDENTITY:
            raise ValueError("5张注册图片不能重复")
        if any(path.suffix.lower() not in IMAGE_TYPES or not path.is_file() for path in resolved):
            raise ValueError("注册图片格式无效")
        # Infer before writing anything: a failed registration never touches the saved registry.
        with self._inference_lock:
            inferred, body_inferred, face_confidences = self._extract_primary_modalities(resolved)
        missing = [path.name for path, feature in zip(resolved, inferred) if feature is None]
        if missing:
            raise ValueError(f"以下注册图未检测到猫脸：{'、'.join(missing)}")
        low_confidence = [
            f"{path.name}（{confidence:.2f}）"
            for path, confidence in zip(resolved, face_confidences)
            if confidence < REGISTRATION_FACE_CONFIDENCE
        ]
        if low_confidence:
            raise ValueError(
                f"以下注册图猫脸置信度低于 {REGISTRATION_FACE_CONFIDENCE:.2f}："
                f"{'、'.join(low_confidence)}"
            )
        vectors_to_save = np.stack(inferred).astype(np.float32)  # type: ignore[arg-type]
        safe = self._safe_id(identity)
        target_images = REGISTRY_IMAGES / safe
        target_vector = REGISTRY_VECTORS / f"{safe}.npy"
        temporary_images = REGISTRY_IMAGES / f".{safe}.tmp"
        if temporary_images.exists():
            shutil.rmtree(temporary_images)
        temporary_images.mkdir(parents=True, exist_ok=False)
        try:
            for index, source in enumerate(resolved, start=1):
                target = temporary_images / f"{index:02d}{source.suffix.lower()}"
                shutil.copy2(source, target)
            REGISTRY_VECTORS.mkdir(parents=True, exist_ok=True)
            temp_vector = REGISTRY_VECTORS / f".{safe}.tmp.npy"
            np.save(temp_vector, vectors_to_save)
            if target_images.exists():
                shutil.rmtree(target_images)
            os.replace(temporary_images, target_images)
            os.replace(temp_vector, target_vector)
        except Exception:
            if temporary_images.exists():
                shutil.rmtree(temporary_images)
            raise
        manifest = self._manifest()
        manifest["identities"] = [entry for entry in manifest["identities"] if entry.get("identity") != identity]
        relative_images = [str(path.relative_to(REGISTRY_ROOT)) for path in sorted(target_images.iterdir())]
        manifest["identities"].append({"identity": identity, "safe_id": safe, "image_files": relative_images, "vector_file": str(target_vector.relative_to(REGISTRY_ROOT)), "created_at_unix": time.time(), "source_hashes": [sha256_file(path) for path in resolved]})
        self._write_manifest(manifest)
        with self._inference_lock:
            with self._gallery_lock:
                self.gallery = [item for item in self.gallery if not (item.source == "user" and item.identity == identity)]
                vectors = np.load(target_vector).astype(np.float32)
                self.gallery.extend(
                    GalleryItem(identity, path, vectors[index], "user", body_inferred[index])
                    for index, path in enumerate(sorted(target_images.iterdir()))
                )
                self._rebuild_templates()

    @staticmethod
    def _associate_subjects(
        bodies: list[DetectedBody], faces: list[DetectedFace],
    ) -> list[tuple[DetectedBody | None, DetectedFace | None]]:
        """Associate each detected face with the smallest body box containing its center."""
        body_faces: dict[int, list[int]] = defaultdict(list)
        assigned_faces: set[int] = set()
        for face_index, face in enumerate(faces):
            center_x = (face.box[0] + face.box[2]) / 2
            center_y = (face.box[1] + face.box[3]) / 2
            containing = [
                index for index, body in enumerate(bodies)
                if body.box[0] <= center_x <= body.box[2] and body.box[1] <= center_y <= body.box[3]
            ]
            if containing:
                body_index = min(
                    containing,
                    key=lambda index: (bodies[index].box[2] - bodies[index].box[0])
                    * (bodies[index].box[3] - bodies[index].box[1]),
                )
                body_faces[body_index].append(face_index)
                assigned_faces.add(face_index)
        subjects: list[tuple[DetectedBody | None, DetectedFace | None]] = []
        for body_index, body in enumerate(bodies):
            candidates = body_faces.get(body_index, [])
            face = max((faces[index] for index in candidates), key=lambda item: item.confidence, default=None)
            subjects.append((body, face))
        subjects.extend((None, face) for index, face in enumerate(faces) if index not in assigned_faces)
        return subjects

    def match_modalities(
        self,
        face_features: list[np.ndarray | None],
        body_features: list[np.ndarray | None],
        face_areas: list[float],
        adaptive: bool = True,
    ) -> list[MatchOutcome]:
        """Late-fuse face/body cosine scores with fixed 62/38 weights."""
        with self._gallery_lock:
            face_templates = self._template_matrix.copy()
            body_templates = self._body_template_matrix.copy()
            names = list(self._template_names)
            gallery = list(self.gallery)
        if not names:
            raise RuntimeError("当前没有可用注册图库")
        return [
            self._match_outcome(
                face, body, area, adaptive, names,
                face_templates, body_templates, gallery,
            )
            for face, body, area in zip(face_features, body_features, face_areas)
        ]

    @staticmethod
    def _branch_scores(
        feature: np.ndarray | None, templates: np.ndarray,
    ) -> np.ndarray | None:
        if feature is None:
            return None
        available = ~np.isnan(templates).any(axis=1)
        scores = np.full(len(templates), -np.inf, dtype=np.float32)
        scores[available] = templates[available] @ feature.astype(np.float32)
        return scores

    @staticmethod
    def _fuse_scores(
        face_scores: np.ndarray | None, body_scores: np.ndarray | None,
    ) -> np.ndarray:
        if face_scores is None:
            return body_scores.copy()  # type: ignore[union-attr]
        if body_scores is None:
            return face_scores.copy()
        face_ok, body_ok = np.isfinite(face_scores), np.isfinite(body_scores)
        scores = np.full(face_scores.shape, -np.inf, dtype=np.float32)
        scores[face_ok & ~body_ok] = face_scores[face_ok & ~body_ok]
        scores[body_ok & ~face_ok] = body_scores[body_ok & ~face_ok]
        both = face_ok & body_ok
        scores[both] = FACE_WEIGHT * face_scores[both] + BODY_WEIGHT * body_scores[both]
        return scores

    @staticmethod
    def _nearest_gallery_path(
        identity: str, gallery: list[GalleryItem],
        face_feature: np.ndarray | None, body_feature: np.ndarray | None,
    ) -> Path | None:
        nearest_path, nearest_score = None, -math.inf
        for item in (item for item in gallery if item.identity == identity):
            values, weights = [], []
            if face_feature is not None:
                values.append(float(item.feature @ face_feature)); weights.append(FACE_WEIGHT)
            if body_feature is not None and item.body_feature is not None:
                values.append(float(item.body_feature @ body_feature)); weights.append(BODY_WEIGHT)
            score = sum(v * w for v, w in zip(values, weights)) / sum(weights) if weights else -math.inf
            if score > nearest_score:
                nearest_path, nearest_score = item.path, score
        return nearest_path

    def _match_outcome(
        self, face_feature: np.ndarray | None, body_feature: np.ndarray | None,
        face_area: float, adaptive: bool, names: list[str],
        face_templates: np.ndarray, body_templates: np.ndarray,
        gallery: list[GalleryItem],
    ) -> MatchOutcome:
        face_scores = self._branch_scores(face_feature, face_templates)
        body_scores = self._branch_scores(body_feature, body_templates)
        scores = self._fuse_scores(face_scores, body_scores)
        order = np.argsort(-scores)
        best_index = int(order[0])
        second_index = int(order[1]) if len(order) > 1 else best_index
        score = float(scores[best_index])
        second_score = float(scores[second_index]) if len(order) > 1 else -1.0
        decision_margin = (
            float(scores[best_index] - scores[second_index])
            if len(order) > 1 else math.inf
        )
        threshold = self._adaptive_threshold(face_area) if adaptive and face_feature is not None else FIXED_MATCH_THRESHOLD
        return MatchOutcome(
            identity=names[best_index], score=score,
            second_identity=names[second_index] if len(order) > 1 else "none",
            second_score=second_score, threshold=threshold, face_area=face_area,
            passed=score >= threshold and (len(order) == 1 or decision_margin >= MATCH_MARGIN),
            match_path=self._nearest_gallery_path(names[best_index], gallery, face_feature, body_feature),
            decision_margin=decision_margin,
        )

    def identify_frame(self, frame_bgr: np.ndarray, adaptive: bool = True) -> tuple[list[FrameMatch], Timing]:
        """对摄像头一帧做多猫识别：返回全部匹配结果与分阶段耗时。"""
        return self._identify_bgr(frame_bgr, adaptive=adaptive)

    def identify_image(self, path: Path, adaptive: bool = True) -> tuple[Image.Image, list[FrameMatch], Timing]:
        """识别静态照片中的全部猫，并返回带身份标注的原图。"""
        decode_started = time.perf_counter()
        rgb = load_rgb(path)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        frame_bgr = rgb[:, :, ::-1].copy()
        matches, timing = self._identify_bgr(frame_bgr, adaptive=adaptive)
        timing = replace(timing, decode_ms=decode_ms)
        if not matches:
            raise ValueError("未检测到猫脸，请选择猫脸更清晰的图片")
        height, width = frame_bgr.shape[:2]
        scale = min(1.0, PHOTO_PREVIEW_WIDTH / width, PHOTO_PREVIEW_HEIGHT / height)
        if scale < 1.0:
            display_frame = cv2.resize(
                frame_bgr, (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            display_matches = [
                self._scale_match(match, scale) for match in matches
            ]
        else:
            display_frame, display_matches = frame_bgr, matches
        return self.annotate_frame(display_frame, display_matches), matches, timing

    def _identify_bgr(
        self, frame_bgr: np.ndarray, adaptive: bool,
    ) -> tuple[list[FrameMatch], Timing]:
        """Run detection, association, batched embeddings and fusion."""
        if self.detector is None or self.model is None:
            raise RuntimeError("模型尚未加载")
        if not self.gallery:
            raise RuntimeError("当前没有可用注册图库")
        with self._inference_lock:
            return self._run_pipeline(frame_bgr, adaptive)

    def _run_pipeline(
        self, frame_bgr: np.ndarray, adaptive: bool,
    ) -> tuple[list[FrameMatch], Timing]:
        started = time.perf_counter()
        face_groups, body_groups, _, _, detect_ms = (
            self._detect_modalities([frame_bgr])
        )
        faces = face_groups[0]
        bodies = body_groups[0]
        association_started = time.perf_counter()
        subjects = self._associate_subjects(bodies, faces)
        association_ms = (time.perf_counter() - association_started) * 1000.0
        if not subjects:
            return [], Timing(
                total_ms=(time.perf_counter() - started) * 1000.0,
                detect_ms=detect_ms, association_ms=association_ms,
            )
        face_features, body_features, face_feature_ms, body_feature_ms = (
            self._extract_subject_features(subjects)
        )
        fusion_started = time.perf_counter()
        outcomes = self.match_modalities(
            face_features, body_features,
            [face.face_area if face is not None else 0.0 for _, face in subjects],
            adaptive=adaptive,
        )
        fusion_ms = (time.perf_counter() - fusion_started) * 1000.0
        matches = self._build_frame_matches(subjects, outcomes)
        total_ms = (time.perf_counter() - started) * 1000.0
        return matches, Timing(
            total_ms=total_ms, detect_ms=detect_ms,
            feature_ms=face_feature_ms + body_feature_ms, match_ms=fusion_ms,
            association_ms=association_ms, face_feature_ms=face_feature_ms,
            body_feature_ms=body_feature_ms, fusion_ms=fusion_ms,
        )

    def _extract_subject_features(
        self, subjects: list[tuple[DetectedBody | None, DetectedFace | None]],
    ) -> tuple[list[np.ndarray | None], list[np.ndarray | None], float, float]:
        face_crops = [face.crop for _, face in subjects if face is not None]
        body_crops = [body.crop for body, _ in subjects if body is not None]
        face_started = time.perf_counter()
        extracted_faces = iter(self._embed_crops(face_crops, flip_tta=False))
        face_feature_ms = (time.perf_counter() - face_started) * 1000.0
        body_started = time.perf_counter()
        extracted_bodies = iter(self._embed_crops(body_crops, flip_tta=False, model=self.body_model))
        body_feature_ms = (time.perf_counter() - body_started) * 1000.0
        face_features = [next(extracted_faces) if face is not None else None for _, face in subjects]
        body_features = [next(extracted_bodies) if body is not None else None for body, _ in subjects]
        return face_features, body_features, face_feature_ms, body_feature_ms

    @staticmethod
    def _build_frame_matches(
        subjects: list[tuple[DetectedBody | None, DetectedFace | None]],
        outcomes: list[MatchOutcome],
    ) -> list[FrameMatch]:
        return [
            FrameMatch(
                box=body.box if body is not None else face.box,
                crop=body.crop if body is not None else face.crop,
                confidence=body.confidence if body is not None else face.confidence,
                identity=outcome.identity,
                score=outcome.score,
                match_path=outcome.match_path,
                second_identity=outcome.second_identity,
                second_score=outcome.second_score,
                threshold=outcome.threshold,
                passed=outcome.passed,
                decision_margin=outcome.decision_margin,
                face_box=face.box if face is not None else None,
                body_box=body.box if body is not None else None,
            )
            for (body, face), outcome in zip(subjects, outcomes)
        ]

    def annotate_frame(self, frame_bgr: np.ndarray, matches: list[FrameMatch]) -> Image.Image:
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        font_size = max(18, min(image.size) // 35)
        font = self._annotation_fonts.get(font_size)
        if font is None:
            try:
                font = ImageFont.truetype(UI_FONT_PATH, font_size)
            except OSError:
                font = ImageFont.load_default()
            self._annotation_fonts[font_size] = font
        line_width = max(3, min(image.size) // 180)
        for index, match in enumerate(matches, start=1):
            label = f"猫 {index} · {match.identity} · 置信度 {match.score:.2f}"
            # 双 ROI：身框绿色粗框 + 小标签，脸框粉色粗框 + 身份标签；
            # 无脸框时身份标签落到主框（即身框）上。
            if match.body_box is not None:
                draw.rectangle(match.body_box, outline=BODY_BOX, width=line_width)
                self._draw_tag(draw, match.body_box, "身体", font, line_width)
            if match.face_box is not None:
                draw.rectangle(match.face_box, outline=PINK, width=line_width)
                self._draw_tag(draw, match.face_box, label, font, line_width)
            else:
                self._draw_tag(draw, match.body_box or match.box, label, font, line_width)
        return image

    @staticmethod
    def _draw_tag(
        draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
        text: str, font, line_width: int,
    ) -> None:
        x1, y1, _, _ = box
        text_box = draw.textbbox((x1, max(2, y1 - font.size - 10)), text, font=font)
        draw.rectangle((text_box[0] - 4, text_box[1] - 3, text_box[2] + 4, text_box[3] + 3), fill=TEXT)
        draw.text((text_box[0], text_box[1]), text, font=font, fill=FIELD)

    @staticmethod
    def _scale_match(match: FrameMatch, scale: float) -> FrameMatch:
        """按预览缩放比例同步缩放主框、脸框与身框。"""
        def scale_box(box: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
            if box is None:
                return None
            return tuple(round(value * scale) for value in box)
        return replace(
            match,
            box=scale_box(match.box),
            face_box=scale_box(match.face_box),
            body_box=scale_box(match.body_box),
        )

    def _rebuild_templates(self) -> None:
        """按身份构建 L2 归一化均值模板（与板端 gallery 建库口径一致）。"""
        names: list[str] = []
        matrix: list[np.ndarray] = []
        body_matrix: list[np.ndarray] = []
        for identity in sorted({item.identity for item in self.gallery}):
            vectors = np.stack([item.feature for item in self.gallery if item.identity == identity]).astype(np.float32)
            template = vectors.mean(axis=0).astype(np.float32)
            template /= max(np.linalg.norm(template), 1e-9)
            names.append(identity)
            matrix.append(template)
            body_vectors = [
                item.body_feature for item in self.gallery
                if item.identity == identity and item.body_feature is not None
            ]
            if body_vectors:
                body_template = np.stack(body_vectors).astype(np.float32).mean(axis=0)
                body_template /= max(np.linalg.norm(body_template), 1e-9)
            else:
                body_template = np.full(DIMENSION, np.nan, dtype=np.float32)
            body_matrix.append(body_template)
        self._template_names = names
        self._template_matrix = np.stack(matrix).astype(np.float32) if matrix else np.zeros((0, DIMENSION), dtype=np.float32)
        self._body_template_matrix = np.stack(body_matrix).astype(np.float32) if body_matrix else np.zeros((0, DIMENSION), dtype=np.float32)
        self._identity_features = {}
        self._identity_body_features = {}
        self._identity_paths = {}
        for identity in names:
            items = [item for item in self.gallery if item.identity == identity]
            self._identity_features[identity] = np.stack([item.feature for item in items]).astype(np.float32)
            available_body = [item.body_feature for item in items if item.body_feature is not None]
            if available_body:
                self._identity_body_features[identity] = np.stack(available_body).astype(np.float32)
            self._identity_paths[identity] = [item.path for item in items]

    @staticmethod
    def _adaptive_threshold(face_area: float) -> float:
        """按脸框面积占比分档，与板端 adaptive_match_threshold 同构。"""
        if face_area >= LARGE_FACE_AREA:
            return LARGE_FACE_THRESHOLD
        if face_area >= MEDIUM_FACE_AREA:
            return MEDIUM_FACE_THRESHOLD
        return SMALL_FACE_THRESHOLD

class CameraPage:
    """摄像头实时多猫识别页，复用“测试图片”页的配色与圆角卡片风格。

    画面线程负责抓帧、YOLO 检测与身份匹配（带节流），Tk 主线程只做
    画面刷新与结果卡片更新，避免 GPU 推理阻塞界面。
    """

    PREVIEW_MIN = 0.30  # 两次识别之间的最小间隔（秒）
    LOCK_FRAMES = 5  # 连续 N 帧识别到同一身份则锁定显示，后续帧波动不消失

    def __init__(self, parent: tk.Frame, service: CatFaceService) -> None:
        self.root = parent.winfo_toplevel()
        self.parent = parent
        self.service = service
        self.messagebox = messagebox
        self.running = False
        self._visible = False
        self.camera = None
        self._thread: threading.Thread | None = None
        self._camera_index_value = 0
        self.camera_index = tk.IntVar(value=0)
        self.adaptive_var = tk.BooleanVar(value=True)
        self._adaptive = True
        self._annotated: Image.Image | None = None
        self._shown_image: Image.Image | None = None
        self._latest_frame_bgr: np.ndarray | None = None
        self._matches: list[FrameMatch] = []
        self._revision = 0
        self._last_revision = -1
        self._last_detect = 0.0
        self._last_timing: Timing | None = None
        self._fps = 0.0  # 最近 30 帧滑动窗口平均帧率
        self._fps_times: deque[float] = deque(maxlen=30)
        self._camera_on = False  # 区分"未开摄像头"与"开了但没检测到猫"
        self._votes: dict[str, int] = {}  # 各身份连续出现帧数（多帧锁定）
        self._locked_ids: set[str] = set()  # 已锁定的身份
        self._last_fingerprint: tuple | None = None  # 结果指纹：无变化不重建面板
        self._tracker = LKBoxTracker()
        self.status_var = tk.StringVar(value="摄像头未开启")
        self._build()
        self.set_enabled(False)

    def set_enabled(self, enabled: bool) -> None:
        self.start_button.set_enabled(enabled)
        self.snapshot_button.set_enabled(enabled)
        self.camera_index_spin.configure(state="normal" if enabled else "disabled")
        self.adaptive_check.configure(state="normal" if enabled else "disabled")

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if visible and self.running:
            self._poll()

    def shutdown(self) -> None:
        self._visible = False
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def _build(self) -> None:
        parent = self.parent
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        left = tk.Frame(parent, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = tk.Frame(parent, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        preview = self._frame(left, "实时画面（多猫同框识别）")
        preview.pack(fill="both", expand=True, pady=(0, 10))
        self.preview_card = RoundedImageCard(
            preview, width=640, height=460, min_w=320, min_h=260,
            resample=Image.Resampling.BILINEAR, allow_upscale=False,
        )
        self.preview_card.pack(fill="both", expand=True)

        controls = tk.Frame(left, bg=BG)
        controls.pack(fill="x")
        tk.Label(controls, text="摄像头索引", bg=BG, fg=TEXT, font=ui_font(12)).pack(side="left", padx=(0, 6))
        self.camera_index_spin = tk.Spinbox(controls, from_=0, to=3, textvariable=self.camera_index, width=4, bg=FIELD, fg=TEXT, buttonbackground=FIELD, relief="flat", font=ui_font(12))
        self.camera_index_spin.pack(side="left", padx=(0, 10))
        self.start_button = RoundedButton(controls, "开启摄像头", self._toggle)
        self.start_button.pack(side="left", padx=(0, 8))
        self.snapshot_button = RoundedButton(controls, "保存快照", self._save_snapshot)
        self.snapshot_button.pack(side="left", padx=(0, 10))
        self.adaptive_check = tk.Checkbutton(controls, text="动态置信度", variable=self.adaptive_var, command=self._sync_adaptive, bg=BG, fg=TEXT, activebackground=BG, activeforeground=TEXT, selectcolor=FIELD, font=ui_font(11))
        self.adaptive_check.pack(side="left")
        tk.Label(left, textvariable=self.status_var, bg=FIELD, fg=TEXT, anchor="w", justify="left", padx=9, pady=8, highlightbackground=FIELD, highlightthickness=2, font=ui_font(11)).pack(fill="x", pady=(8, 0))

        results = self._frame(right, "识别到的猫（名字 + 照片）")
        results.pack(fill="both", expand=True)
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=1)
        panel = tk.Frame(results, bg=BG)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)
        canvas = tk.Canvas(panel, bg=BG, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)
        self.cards_frame = tk.Frame(canvas, bg=BG)
        window = canvas.create_window((0, 0), window=self.cards_frame, anchor="nw")
        self.cards_frame.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda _event: canvas.itemconfigure(window, width=canvas.winfo_width()))
        self.summary_label = tk.Label(results, text="当前画面：0 只猫", bg=PINK, fg=TEXT, anchor="w", font=ui_font(14, bold=True), padx=12, pady=10)
        self.summary_label.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self._rebuild_cards()

    def _frame(self, parent, title):
        return tk.LabelFrame(parent, text=title, bg=BG, fg=TEXT, font=ui_font(13, bold=True), bd=2, relief="groove", padx=12, pady=12)

    def _sync_adaptive(self) -> None:
        self._adaptive = bool(self.adaptive_var.get())

    def _toggle(self) -> None:
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        if self.running or (self._thread is not None and self._thread.is_alive()):
            return
        if self.service.model is None:
            self.messagebox.showinfo("模型未就绪", "请等待模型加载完成后再开启摄像头")
            return
        self.running = True
        self._camera_index_value = self.camera_index.get()
        self.status_var.set("正在打开摄像头…")
        self.start_button.set_text("关闭摄像头")
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        if self._visible:
            self._poll()

    def _stop(self) -> None:
        if not self.running:
            return
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._reset_view("摄像头已关闭")

    def _camera_failed(self, index: int) -> None:
        self._reset_view("摄像头未开启")
        self.messagebox.showerror("无法打开摄像头", f"无法打开摄像头 {index}，请检查是否被占用或已连接。")

    def _thread_end(self) -> None:
        self._reset_view("摄像头已断开或读取失败")

    def _reset_view(self, status: str) -> None:
        self._thread = None
        self._camera_on = False
        self._fps = 0.0
        self._fps_times.clear()
        self._votes = {}
        self._locked_ids = set()
        self._last_fingerprint = None
        self.start_button.set_text("开启摄像头")
        self.status_var.set(status)
        self._matches = []
        self._revision += 1
        self._annotated = None
        self._shown_image = None
        self._latest_frame_bgr = None
        self._tracker.reset()
        self.preview_card.set_image(Image.new("RGB", (16, 16), FIELD))
        self._rebuild_cards()

    def _capture_loop(self) -> None:
        index = self._camera_index_value
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            self.running = False
            self.root.after(0, lambda: self._camera_failed(index))
            return
        # 请求更高采集分辨率：小脸像素更多，YOLO 置信度更稳
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.camera = cap
        self._camera_on = True
        self.root.after(0, lambda: self.status_var.set(
            f"摄像头 {index} 已开启（{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}）"))
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                self._latest_frame_bgr = frame
                now = time.time()
                detections = None
                if now - self._last_detect >= self.PREVIEW_MIN:
                    self._last_detect = now
                    try:
                        detections, timing = self.service.identify_frame(frame, adaptive=self._adaptive)
                        self._last_timing = timing
                    except Exception as error:
                        self.root.after(0, lambda captured=error: self.status_var.set(f"识别出错：{captured}"))
                        detections, timing = [], Timing(total_ms=0.0)
                        self._last_timing = timing
                matches = self._tracker.step(frame, detections)
                self._matches = matches
                # 只有通过动态阈值的身份才能参与多帧锁定，避免低置信误报。
                present = {match.identity for match in matches if match.passed}
                self._votes = {ident: self._votes.get(ident, 0) + 1 for ident in present}
                self._locked_ids = {ident for ident, count in self._votes.items() if count >= self.LOCK_FRAMES}
                self._revision += 1
                self._annotated = self._draw_boxes(frame, matches)
        finally:
            cap.release()
            self.camera = None
            self._thread = None
            if self.running:
                self.running = False
                self.root.after(0, self._thread_end)

    def _draw_boxes(self, frame: np.ndarray, matches: list[FrameMatch]) -> Image.Image:
        height, width = frame.shape[:2]
        scale = min(1.0, CAMERA_PREVIEW_WIDTH / width, CAMERA_PREVIEW_HEIGHT / height)
        if scale >= 1.0:
            return self.service.annotate_frame(frame, matches)
        preview = cv2.resize(
            frame, (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        scaled_matches = [
            self.service._scale_match(match, scale) for match in matches
        ]
        return self.service.annotate_frame(preview, scaled_matches)

    def _poll(self) -> None:
        if not self.running or not self._visible:
            return
        try:
            if self._annotated is not None and self._annotated is not self._shown_image:
                self._shown_image = self._annotated
                self.preview_card.set_image(self._annotated)
            if self._last_revision != self._revision:
                self._last_revision = self._revision
                self._rebuild_cards()
        except Exception as error:  # 渲染异常绝不能中断刷新链
            self.status_var.set(f"刷新异常：{error}")
        self.root.after(30, self._poll)

    def _rebuild_cards(self) -> None:
        matches = self._matches
        timing = self._last_timing
        # 摘要行（含"检测→匹配"耗时与帧率）每帧刷新；结果指纹只认"画面里
        # 是谁"：置信度分数每帧微变，若进指纹会导致面板每帧重建。身份集合
        # 不变就完全静止，卡片上的分数/耗时/帧率只随重建时刷新。
        self._fps_times.append(time.time())
        if len(self._fps_times) >= 2:
            span = self._fps_times[-1] - self._fps_times[0]
            self._fps = (len(self._fps_times) - 1) / span if span > 0 else 0.0
        timing_text = f" · 检测到匹配耗时 {timing.total_ms:.0f} ms" if timing is not None else ""
        fps_text = f" · 帧率 {self._fps:.1f} FPS" if self._fps > 0 else ""
        self.summary_label.configure(text=f"当前画面：{len(matches)} 只猫{timing_text}{fps_text}")
        # 必须先于 destroy 判断，否则稳定状态下每次刷新会把面板清空。
        fingerprint = tuple((m.identity, m.passed, m.box) for m in matches)
        if fingerprint == self._last_fingerprint:
            return
        self._last_fingerprint = fingerprint
        for child in self.cards_frame.winfo_children():
            child.destroy()
        if not matches:
            if self._camera_on:
                tk.Label(self.cards_frame, text="摄像头已开启，但画面中未检测到猫\n（检测置信度低于 0.35）\n请让猫靠近镜头、正对脸、保证光线", bg=BG, fg="#B04A2A", font=ui_font(12)).pack(anchor="w", pady=8)
            else:
                tk.Label(self.cards_frame, text="摄像头未开启", bg=BG, fg=TEXT, font=ui_font(12)).pack(anchor="w", pady=8)
            return
        # 锁定的身份优先显示
        matches = sorted(matches, key=lambda m: not (m.passed and m.identity in self._locked_ids))
        for index, match in enumerate(matches, start=1):
            row = tk.Frame(self.cards_frame, bg=BG)
            row.pack(fill="x", pady=4)
            card = RoundedImageCard(row, width=84, height=84, min_w=56, min_h=56)
            card.pack(side="left", padx=(0, 10))
            card.set_image(match.crop)
            info = tk.Frame(row, bg=BG)
            info.pack(side="left", fill="both", expand=True)
            # 只显示姓名、置信度与"检测→匹配"耗时
            tk.Label(info, text=f"猫 {index} · {match.identity}", bg=BG, fg=TEXT, font=ui_font(15, bold=True), anchor="w").pack(fill="x", pady=(6, 0))
            tk.Label(info, text=f"置信度：{match.score:.4f}", bg=BG, fg=TEXT, font=ui_font(12), anchor="w").pack(fill="x")
            if timing is not None:
                tk.Label(info, text=timing_summary(timing), bg=BG, fg="#B04A2A", font=ui_font(11), anchor="w").pack(fill="x")
            if self._fps > 0:
                tk.Label(info, text=f"帧率：{self._fps:.1f} FPS", bg=BG, fg=TEXT, font=ui_font(11), anchor="w").pack(fill="x")

    def _save_snapshot(self) -> None:
        if self._latest_frame_bgr is None:
            self.messagebox.showinfo("无画面", "摄像头未开启或尚无可用画面")
            return
        out = ROOT / "snapshots"
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        frame_name = f"frame_{stamp}.png"
        self.service.annotate_frame(self._latest_frame_bgr, self._matches).save(out / frame_name)
        saved = [frame_name]
        for index, match in enumerate(self._matches, start=1):
            name = re.sub(r"[^\w-]+", "_", match.identity, flags=re.UNICODE).strip("_") or "cat"
            crop_name = f"{stamp}_cat{index}_{name}.png"
            match.crop.save(out / crop_name)
            saved.append(crop_name)
        self.status_var.set(f"快照已保存到 {out}：{'、'.join(saved)}")


class CatFaceApp:
    def __init__(self, root: "tk.Tk") -> None:
        self.tk = tk
        self.filedialog, self.messagebox = filedialog, messagebox
        self.root = root
        self.service = CatFaceService()
        self.worker = BackgroundWorker()
        self.selected_registration: list[Path] = []
        self.header_art: ImageTk.PhotoImage | None = None
        self.busy = False
        root.title("猫猫识别器")
        root.configure(bg=BG)
        root.minsize(1560, 800)
        self.status = tk.StringVar(value="正在加载 YOLO11n 与 MobileOne-S1…")
        self.identity_var = tk.StringVar()
        self._build()
        self._set_enabled(False)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._run_async("加载模型和默认注册图库", self.service.load, self._loaded)

    def _frame(self, parent, title: str):
        frame = self.tk.LabelFrame(parent, text=title, bg=BG, fg=TEXT, font=ui_font(13, bold=True), bd=2, relief="groove", padx=12, pady=12)
        return frame

    def _build(self) -> None:
        tk = self.tk
        header = tk.Frame(self.root, bg=BG, height=250)
        header.pack(fill="x")
        header.pack_propagate(False)
        title_label = tk.Label(header, text="猫猫识别器", bg=BG, fg=TEXT, font=ui_font(39), bd=0, highlightthickness=0)
        title_label.place(relx=0.5, rely=0.5, anchor="center")
        if HEADER_ART_PATH.is_file():
            with Image.open(HEADER_ART_PATH) as image:
                art = ImageOps.exif_transpose(image).convert("RGBA")
                alpha_box = art.getchannel("A").getbbox()
                if alpha_box is not None:
                    art = art.crop(alpha_box)
                art.thumbnail((500, 288), Image.Resampling.LANCZOS)
                art = art.resize((432, round(art.height * 0.8)), Image.Resampling.LANCZOS)
                self.header_art = ImageTk.PhotoImage(art)
            art_label = tk.Label(header, image=self.header_art, bg=BG, bd=0, highlightthickness=0)
            art_label.place(in_=title_label, relx=1.0, x=60, rely=0.5, anchor="w")
        nav = tk.Frame(self.root, bg=BG)
        nav.pack(fill="x", padx=16)
        self.test_tab_button = RoundedButton(nav, "图片多猫识别", lambda: self._show_page("test"))
        self.test_tab_button.pack(side="left", padx=(0, 8))
        self.camera_tab_button = RoundedButton(nav, "摄像头多猫识别", lambda: self._show_page("camera"))
        self.camera_tab_button.pack(side="left")
        body = tk.Frame(self.root, bg=BG, padx=16, pady=4)
        body.pack(fill="both", expand=True)
        self.page_holder = tk.Frame(body, bg=BG)
        self.page_holder.pack(fill="both", expand=True)
        self.page_test = tk.Frame(self.page_holder, bg=BG)
        self.page_camera = tk.Frame(self.page_holder, bg=BG)
        self.camera_page = CameraPage(self.page_camera, self.service)
        self._build_page_test()
        status = tk.Label(self.root, textvariable=self.status, bg=PINK, fg=TEXT, anchor="w", font=ui_font(12), padx=16, pady=9)
        status.pack(fill="x", side="bottom")
        self._show_page("test")

    def _show_page(self, name: str) -> None:
        self.test_tab_button.set_active(name == "test")
        self.camera_tab_button.set_active(name == "camera")
        for page in (self.page_test, self.page_camera):
            page.pack_forget()
        (self.page_test if name == "test" else self.page_camera).pack(fill="both", expand=True)
        self.camera_page.set_visible(name == "camera")

    def _on_close(self) -> None:
        self.camera_page.shutdown()
        self.service.close()
        self.root.destroy()

    def _build_page_test(self) -> None:
        tk = self.tk
        body = self.page_test
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        gallery = self._frame(left, "注册身份库")
        gallery.pack(fill="both", expand=True, pady=(0, 10))
        self.gallery_text = tk.Text(gallery, height=16, bg=FIELD, fg=TEXT, font=ui_font(12), relief="flat", highlightbackground=FIELD, highlightcolor=FIELD, highlightthickness=2, wrap="word", state="disabled")
        self.gallery_text.pack(fill="both", expand=True)

        registration = self._frame(left, "新增 / 覆盖身份注册（恰好5张）")
        registration.pack(fill="x")
        tk.Label(registration, text="身份 ID", bg=BG, fg=TEXT, font=ui_font(12)).pack(anchor="w")
        self.identity_entry = tk.Entry(registration, textvariable=self.identity_var, bg=FIELD, fg=TEXT, insertbackground=TEXT, relief="flat", highlightbackground=FIELD, highlightcolor=FIELD, highlightthickness=2, font=ui_font(12))
        self.identity_entry.pack(fill="x", pady=(2, 8))
        self.choose_registration_button = RoundedButton(registration, "选择5张注册图片", self._choose_registration)
        self.choose_registration_button.pack(fill="x")
        self.registration_label = tk.Label(registration, text="尚未选择图片", bg=FIELD, fg=TEXT, justify="left", anchor="w", highlightbackground=FIELD, highlightthickness=2, padx=8, pady=7, wraplength=330, font=ui_font(11))
        self.registration_label.pack(fill="x", pady=6)
        self.register_button = RoundedButton(registration, "保存注册身份", self._register)
        self.register_button.pack(fill="x")

        actions = self._frame(right, "测试图片")
        actions.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        action_row = tk.Frame(actions, bg=BG)
        action_row.pack(fill="x")
        self.random_button = RoundedButton(action_row, "随机测试默认样本", self._random_test)
        self.random_button.pack(side="left", padx=(0, 8))
        self.choose_test_button = RoundedButton(action_row, "选择一张含单猫/多猫的照片", self._choose_test)
        self.choose_test_button.pack(side="left")
        self.test_path_label = tk.Label(actions, text="请选择或随机抽取一张猫照片", bg=FIELD, fg=TEXT, anchor="w", highlightbackground=FIELD, highlightthickness=2, padx=8, pady=7, font=ui_font(11))
        self.test_path_label.pack(fill="x", pady=(8, 0))

        result = self._frame(right, "多猫识别结果")
        result.grid(row=1, column=0, sticky="nsew")
        right.rowconfigure(1, weight=1)
        result.columnconfigure(0, weight=1)
        result.rowconfigure(0, weight=3)
        result.rowconfigure(1, weight=2)

        query_holder = tk.Frame(result, bg=BG)
        query_holder.grid(row=0, column=0, sticky="nsew")
        query_holder.columnconfigure(0, weight=1)
        query_holder.rowconfigure(1, weight=1)
        tk.Label(query_holder, text="带框原图（从左到右编号）", bg=BG, fg=TEXT, font=ui_font(12, bold=True)).grid(row=0, column=0)
        self.query_label = RoundedImageCard(query_holder, width=700, height=300, min_w=420, min_h=220)
        self.query_label.grid(row=1, column=0, sticky="nsew", pady=(5, 8))

        panel = tk.Frame(result, bg=BG)
        panel.grid(row=1, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)
        canvas = tk.Canvas(panel, bg=BG, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)
        self.photo_cards_frame = tk.Frame(canvas, bg=BG)
        self.photo_cards_frame.columnconfigure(0, weight=1, uniform="photo_result")
        self.photo_cards_frame.columnconfigure(1, weight=1, uniform="photo_result")
        window = canvas.create_window((0, 0), window=self.photo_cards_frame, anchor="nw")
        self.photo_cards_frame.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda _event: canvas.itemconfigure(window, width=canvas.winfo_width()))
        self.result_text = tk.Label(result, text="等待测试", bg=PINK, fg=TEXT, justify="left", anchor="w", font=ui_font(13, bold=True), padx=12, pady=9)
        self.result_text.grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def _set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (getattr(self, "choose_registration_button", None), getattr(self, "register_button", None), getattr(self, "random_button", None), getattr(self, "choose_test_button", None), getattr(self, "identity_entry", None)):
            if widget is not None:
                if isinstance(widget, RoundedButton):
                    widget.set_enabled(enabled)
                else:
                    widget.configure(state=state)
        if getattr(self, "camera_page", None) is not None:
            self.camera_page.set_enabled(enabled)

    def _run_async(self, label: str, work, done) -> None:
        if self.busy:
            return
        self.busy = True
        self._set_enabled(False)
        self.status.set(f"{label}…")

        # 统一投递到常驻工作线程：加载阶段已在该线程完成 CUDA 热身，
        # 后续每次识别都不会再重复支付新线程首次前向的开销。
        self.worker.submit(
            work,
            lambda value: self.root.after(
                0, lambda captured_value=value: self._done(done, captured_value)),
            # Bind the exception as a default argument.  Python clears the
            # ``except ... as error`` variable after the block, so a plain
            # closure would otherwise raise NameError in Tk's callback.
            lambda error: self.root.after(
                0, lambda captured_error=error: self._failed(label, captured_error)),
        )

    def _done(self, callback, value) -> None:
        self.busy = False
        self._set_enabled(True)
        callback(value)

    def _failed(self, label: str, error: Exception) -> None:
        self.busy = False
        self._set_enabled(True)
        self.status.set(f"{label}失败：{error}")
        self.messagebox.showerror("操作失败", str(error))

    def _loaded(self, _value) -> None:
        self._refresh_gallery()
        self.status.set("模型已就绪：可随机测试、选择测试图片或注册新身份")

    def _refresh_gallery(self) -> None:
        defaults = self.service.default_ids
        users = sorted({item.identity for item in self.service.gallery if item.source == "user"})
        text = "默认样本（每只5张）\n" + "\n".join(f"  • {item}" for item in defaults)
        text += f"\n\n用户注册身份（{len(users)}）\n" + ("\n".join(f"  • {item}" for item in users) if users else "  暂无")
        text += f"\n\n当前图库：{len(self.service.gallery)} 张注册照片"
        self.gallery_text.configure(state="normal")
        self.gallery_text.delete("1.0", "end")
        self.gallery_text.insert("1.0", text)
        self.gallery_text.configure(state="disabled")

    def _choose_registration(self) -> None:
        paths = self.filedialog.askopenfilenames(title="选择恰好5张注册图片", filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")])
        if not paths:
            return
        self.selected_registration = [Path(path) for path in paths]
        self.registration_label.configure(text=f"已选择 {len(paths)} 张\n" + "\n".join(path.name for path in self.selected_registration))

    def _register(self) -> None:
        identity = self.identity_var.get().strip()
        if not identity:
            self.messagebox.showwarning("缺少身份 ID", "请先输入身份 ID")
            return
        if len(self.selected_registration) != DEFAULT_IMAGES_PER_IDENTITY:
            self.messagebox.showwarning("注册图片不足", "请恰好选择5张注册图片")
            return
        existing_users = {item.identity for item in self.service.gallery if item.source == "user"}
        if identity in existing_users and not self.messagebox.askyesno("覆盖确认", f"“{identity}”已存在，是否用这5张新图覆盖？"):
            return
        self._run_async("注册身份", lambda: self.service.register(identity, self.selected_registration), self._registered)

    def _registered(self, _value) -> None:
        self.identity_var.set("")
        self.selected_registration = []
        self.registration_label.configure(text="注册成功，已清空图片选择")
        self._refresh_gallery()
        self.status.set("注册成功，身份与向量已持久保存")

    def _random_test(self) -> None:
        item = random.choice(self.service.default_gallery)
        self._test_path(item.path, "随机默认样本")

    def _choose_test(self) -> None:
        path = self.filedialog.askopenfilename(title="选择一张测试图片", filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")])
        if path:
            self._test_path(Path(path), "用户选择")

    def _test_path(self, path: Path, source: str) -> None:
        self.test_path_label.configure(text=f"{source}：{path}")
        self._run_async("识别照片中的全部猫脸", lambda: self.service.identify_image(path), self._identified)

    def _identified(self, value) -> None:
        annotated, matches, timing = value
        self.query_label.set_image(annotated)
        for child in self.photo_cards_frame.winfo_children():
            child.destroy()
        for index, match in enumerate(matches, start=1):
            card_row, card_column = divmod(index - 1, 2)
            row = tk.Frame(self.photo_cards_frame, bg=FIELD, padx=8, pady=7)
            row.grid(row=card_row, column=card_column, sticky="nsew", padx=4, pady=4)
            header = tk.Frame(row, bg=FIELD)
            header.pack(fill="x")
            tk.Label(header, text=f"猫 {index}：{match.identity}", bg=FIELD, fg=TEXT, font=ui_font(15, bold=True), anchor="w").pack(fill="x")
            photos = tk.Frame(row, bg=FIELD)
            photos.pack(fill="x", pady=(4, 2))
            crop_holder = tk.Frame(photos, bg=FIELD)
            crop_holder.pack(side="left", expand=True)
            tk.Label(crop_holder, text="检测照", bg=FIELD, fg=TEXT, font=ui_font(11, bold=True)).pack()
            crop_card = RoundedImageCard(crop_holder, width=112, height=104, min_w=92, min_h=86)
            crop_card.pack(pady=(3, 0))
            crop_card.set_image(match.crop)

            match_holder = tk.Frame(photos, bg=FIELD)
            match_holder.pack(side="left", expand=True)
            tk.Label(match_holder, text="注册对照照", bg=FIELD, fg=TEXT, font=ui_font(11, bold=True)).pack()
            match_card = RoundedImageCard(match_holder, width=112, height=104, min_w=92, min_h=86)
            match_card.pack(pady=(3, 0))
            if match.match_path is not None:
                with Image.open(match.match_path) as image:
                    match_card.set_image(ImageOps.exif_transpose(image).convert("RGB"))

            tk.Label(
                row,
                text=f"置信度：{match.score:.4f}",
                bg=FIELD, fg=TEXT, justify="left", anchor="w", font=ui_font(12),
            ).pack(fill="x", pady=(3, 0))
        names = "、".join(match.identity for match in matches)
        self.result_text.configure(
            text=(f"检测到 {len(matches)} 只猫：{names}\n"
                  f"{timing_summary(timing)}\n"
                  f"文件读取 {timing.decode_ms:.0f} ms（不计入端到端主耗时）"))
        self.status.set(f"识别完成：已输出 {len(matches)} 只猫的检测照、姓名和注册对照照")


def smoke_test() -> None:
    service = CatFaceService()
    try:
        service.load()
        chosen = random.choice(service.default_gallery)
        _annotated, matches, timing = service.identify_image(chosen.path)
        if not matches or not service.default_ids or not service.gallery:
            raise RuntimeError("GUI smoke test validation failed")
        first = matches[0]
        print(json.dumps({"device": service.device_arg, "deploy_models": service.deploy_models, "detector": "unified_cat_body_face_yolo", "default_identities": service.default_ids, "gallery_identities": sorted({item.identity for item in service.gallery}), "gallery_images": len(service.gallery), "query": str(chosen.path), "detected_cats": len(matches), "predicted_identity": first.identity, "similarity": first.score, "threshold": first.threshold, "passed": first.passed, "timing": timing.__dict__, "feature_dimension": DIMENSION}, ensure_ascii=False, indent=2))
    finally:
        service.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="猫猫识别器桌面 GUI")
    parser.add_argument("--smoke-test", action="store_true", help="不启动界面，验证默认图库和一次识别")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    import tkinter as tk

    enable_high_dpi()
    register_ui_font()
    root = tk.Tk()
    root.tk.call("tk", "scaling", root.winfo_fpixels("1i") / 72.0)
    root.option_add("*Font", ui_font(12))
    CatFaceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
