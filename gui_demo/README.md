# PC 猫身份识别 GUI Demo

本目录是跨平台（Windows / Linux）的独立 GUI 运行包：使用一次双类别 YOLO 检测 `cat_body` 和 `cat_face`，然后串行运行 Face/Body MobileOne-S1，按 0.62/0.38 融合后直接输出 Top-1。

所有代码路径均为**相对路径**（以本文件所在目录为基准），整个仓库拷到任何一台装有依赖的机器即可运行，无需修改任何配置。

## 目录结构

```
gui_demo/
├── run_gui.py            # 入口（GUI + --smoke-test）
├── mobileone_embedder.py # MobileOne-S1 384D 特征模型定义
├── requirements.txt      # Python 依赖
├── models/               # 检测与识别权重（随包自带）
│   ├── cat_body_face_yolo11n_v1.pt
│   ├── face_mobileone_s1.pt
│   └── body_mobileone_s1.pt
├── assets/               # 界面字体与头图（随包自带）
│   ├── ui_font.ttf       # 猫啃什锦黑-轻量版
│   └── header.png
├── teach/                # 算法详解文档（浏览器打开，图片用相对路径）
└── snapshots/            # 摄像头快照（首次保存时自动创建）
../data/gui_registry      # 注册图库（默认/用户注册照片与向量，程序要求存在）
```

依赖关系：`gui_demo` 需要仓库根目录下的 `../data/gui_registry`（注册图库，含 5 只默认猫与用户注册身份），请先新建这个文件夹（含 `data/` 目录）。

## 依赖安装

### 1. 系统依赖

**Linux (Debian/Ubuntu)**：

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-tk \
    libgl1 libglib2.0-0 fontconfig
```

- `python3-tk`：Tkinter 在 Linux 上不随 Python 自带，必须单独装
- `libgl1`、`libglib2.0-0`：OpenCV 运行所需的 GL/GLib 库，缺失会报 `libGL.so.1: cannot open shared object file`
- `fontconfig`（含 `fc-cache`）：程序启动时会自动把自带字体安装到 `~/.local/share/fonts` 并刷新缓存

**Windows**：从 python.org 安装 Python 3.10+（Tkinter 默认包含），无需额外系统依赖。不依赖 ComfyUI 的 Python 环境。

### 2. Python 依赖

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install -r gui_demo/requirements.txt
```

`requirements.txt` 包含：`torch`、`torchvision`、`ultralytics`（YOLO 检测）、`timm`（MobileOne 骨干）、`opencv-python`、`numpy`、`pillow`。

**GPU（CUDA）加速（推荐）**：默认安装的 torch 是 CPU 版。需要 GPU 时，先按 [PyTorch 官方安装向导](https://pytorch.org/get-started/locally/) 选择对应 CUDA 版本的命令安装 torch/torchvision，再装其余依赖：

```bash
# 示例（CUDA 12.x，具体版本以官方向导为准）：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r gui_demo/requirements.txt
```

无 GPU 也能运行（自动回退 CPU，速度较慢）。

## 启动

在仓库根目录下执行（Windows 用 `python`，Linux 同样）：

```bash
python gui_demo/run_gui.py
```

只做模型加载和一次识别（不启动界面，用于验证环境）：

```bash
python gui_demo/run_gui.py --smoke-test
```

## 说明

- 图片页与摄像头页使用同一条低延迟链路：关闭翻转 TTA，不加载 NekoNet/SIFT/XGBoost；Face MobileOne 和 Body MobileOne 串行执行。首次加载会构建注册模板并完成两轮 batch-1 预热，启动耗时不计入单帧推理。
- 真实统一 YOLO 的脸框、身体框和 ROI 可视化位于 `../demo_hisi/examples/unified_yolo_face_body_roi.jpg`（仅文档参考，非程序依赖）。
- `../data/gui_registry` 保存默认/用户注册照片和向量；按用户要求，根目录下的 `data` 数据集整体保留。界面"保存快照"会在本目录按需创建 `snapshots`。
- 摄像头索引从 0 开始；Linux 下若识别不到摄像头，可尝试 `sudo apt install v4l-utils` 后用 `v4l2-ctl --list-devices` 查看设备号。
