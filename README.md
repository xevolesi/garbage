# YuNet PyTorch

A clean PyTorch reimplementation of [YuNet](https://github.com/ShiqiYu/libfacedetection) — a lightweight face detection model with facial landmark prediction.

## Overview

YuNet is designed for real-time face detection on edge devices. This reimplementation provides:

- **Clean, readable code** — No mmdet/mmcv dependencies
- **Modern Python** — Type hints, Pydantic configs
- **Faithful reproduction** — Matches original architecture and training setup

### Model Architecture

```
Input (640×640×3)
       ↓
┌─────────────────┐
│    Backbone     │  Stem + DepthWise blocks with MaxPool
│   (6 stages)    │  Outputs: P8, P16, P32 feature maps
└────────┬────────┘
         ↓
┌─────────────────┐
│    TinyFPN      │  Top-down feature fusion
│   (3 levels)    │  Lateral convs + Upsampling
└────────┬────────┘
         ↓
┌─────────────────┐
│  Detection Head │  Per-level heads for:
│  (×3 levels)    │  - Classification (1 class)
└────────┬────────┘  - Box regression (4 coords)
         ↓           - Objectness (1 score)
    Predictions      - Keypoints (5×2 coords)
```

## Installation

```bash
git clone https://github.com/your-username/yunet_pytorch.git
cd yunet_pytorch
pip install -r requirements.txt
```

## Training

```bash
python train.py --config config.yml
```

### Configuration

Key training parameters in `config.yml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 16 | Original paper uses 16 |
| `lr` | 0.01 | Initial learning rate |
| `epochs` | 640 | Total training epochs |
| `milestones` | [400, 544] | LR decay epochs |
| `warmup_iters` | 1500 | Linear warmup iterations |

## Dataset

Uses [WiderFace](http://shuoyang1213.me/WIDERFACE/) dataset. Prepare as:

```
data/widerface/
├── WIDER_train/images/
├── WIDER_val/images/
└── labelv2/
    ├── train/labelv2.txt
    └── val/labelv2.txt
```

## Results

| Model | Easy | Medium | Hard | Params |
|-------|------|--------|------|--------|
| YuNet-n (original) | 89.1 | 87.2 | 73.0 | ~75K |
| YuNet-n (this repo) | TBD | TBD | TBD | ~75K |

---

# 🚀 Modernization Roadmap (2025-2026)

The original YuNet was published around 2022-2023. Here are suggestions to make it state-of-the-art:

## 1. Replace SimOTA with Task-Aligned Assigner (TAL)

SimOTA is from YOLOX (2021). TAL from RTMDet/YOLOv8 is the current standard:

```python
# TAL key insight: alignment metric = cls_score^α × IoU^β
alignment_metric = (cls_scores ** alpha) * (ious ** beta)
```

**Benefits**: Better matching between classification confidence and localization quality, more stable training.

**Priority**: 🥇 High | **Expected Gain**: +1-2% mAP

---

## 2. Add Distribution Focal Loss (DFL) for Box Regression

Instead of regressing 4 values directly, predict a distribution over discrete bins:

```python
# Instead of: box = [x, y, w, h]  # 4 values
# Predict: box = [dist_x1, dist_x2, dist_y1, dist_y2]  # 4 × 16 bins = 64 values
# Then: x1 = Σ softmax(dist_x1) × bin_values
```

**Benefits**: More accurate localization, especially for small faces.

**Priority**: 🥇 High | **Expected Gain**: +1-2% mAP

---

## 3. Modernize Backbone with Reparameterization

Use **RepVGG-style** blocks that are multi-branch during training but merge to single conv at inference:

```python
class RepConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.conv3x3 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1)
        self.identity = nn.BatchNorm2d(in_ch) if in_ch == out_ch else None
    
    def forward(self, x):
        return self.conv3x3(x) + self.conv1x1(x) + (self.identity(x) if self.identity else 0)
    
    def fuse(self):
        # Merge all branches into single 3x3 conv for inference
        ...
```

**Benefits**: Better accuracy during training, same speed at inference.

**Priority**: 🥈 Medium | **Expected Gain**: Better accuracy, same inference speed

---

## 4. Add Lightweight Attention

Add **Coordinate Attention** or **ECA** (Efficient Channel Attention) after key blocks:

```python
class CoordAttention(nn.Module):
    """Coordinate Attention - captures long-range dependencies efficiently"""
    def __init__(self, channels, reduction=32):
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv = nn.Conv2d(channels, channels // reduction, 1)
        self.conv_h = nn.Conv2d(channels // reduction, channels, 1)
        self.conv_w = nn.Conv2d(channels // reduction, channels, 1)
```

**Benefits**: Better capture of face structure without much overhead.

**Priority**: 🥉 Lower | **Expected Gain**: +0.3-0.5% mAP

---

## 5. Upgrade to Decoupled Head with Shared Stem

Current YOLOv8/v11 style:

```python
class DecoupledHead(nn.Module):
    def __init__(self, in_ch):
        self.stem = DWConv(in_ch, in_ch)  # Shared
        
        # Separate branches
        self.cls_branch = nn.Sequential(DWConv(in_ch, in_ch), nn.Conv2d(in_ch, num_cls, 1))
        self.reg_branch = nn.Sequential(DWConv(in_ch, in_ch), nn.Conv2d(in_ch, 4 * reg_max, 1))  # DFL
        self.kps_branch = nn.Sequential(DWConv(in_ch, in_ch), nn.Conv2d(in_ch, num_kps * 2, 1))
```

**Priority**: 🥈 Medium

---

## 6. Better Keypoint Representation: SimCC

Instead of direct regression, use **SimCC** (Simple Coordinate Classification):

```python
# Instead of regressing (x, y) directly
# Predict x_distribution (1D) and y_distribution (1D) separately
# Much more accurate for keypoints

class SimCCHead(nn.Module):
    def __init__(self, in_ch, num_kps, simcc_split_ratio=2.0):
        self.x_linear = nn.Linear(in_ch, int(width * simcc_split_ratio))
        self.y_linear = nn.Linear(in_ch, int(height * simcc_split_ratio))
```

**Benefits**: State-of-the-art keypoint accuracy with minimal overhead.

**Priority**: 🥈 Medium | **Expected Gain**: +2-3% NME for landmarks

---

## 7. Training Enhancements

| Enhancement | Description | Priority |
|------------|-------------|----------|
| **EMA** | Exponential Moving Average of weights | 🥇 High |
| **Mosaic + MixUp** | Data augmentation for better generalization | 🥉 Lower |
| **Cosine LR** | Replace step LR with cosine annealing | 🥈 Medium |
| **Label Smoothing** | Soft targets for classification | 🥉 Lower |
| **Knowledge Distillation** | Train with a larger teacher model | 🥈 Medium |

```python
# Add EMA
ema = ModelEMA(model, decay=0.9999)
# After each step:
ema.update(model)
# For eval:
ema.ema.eval()
```

---

## 8. Architecture Evolution Summary

```
Current YuNet (2022):          Modernized YuNet (2025):
                               
┌─────────────────┐            ┌─────────────────┐
│  Conv Backbone  │    →       │  RepConv + Attn │
└────────┬────────┘            └────────┬────────┘
         │                              │
┌────────▼────────┐            ┌────────▼────────┐
│    TinyFPN      │    →       │  BiFPN / PAN    │
└────────┬────────┘            └────────┬────────┘
         │                              │
┌────────▼────────┐            ┌────────▼────────┐
│  Coupled Head   │    →       │ Decoupled + DFL │
└────────┬────────┘            └────────┬────────┘
         │                              │
┌────────▼────────┐            ┌────────▼────────┐
│     SimOTA      │    →       │      TAL        │
└─────────────────┘            └─────────────────┘
```

---

## 9. Implementation Priority

| Priority | Change | Expected Gain |
|----------|--------|---------------|
| 🥇 High | TAL + DFL | +1-2% mAP |
| 🥇 High | EMA | +0.5-1% mAP, training stability |
| 🥈 Medium | RepConv backbone | Better accuracy, same inference speed |
| 🥈 Medium | SimCC keypoints | +2-3% NME for landmarks |
| 🥈 Medium | Cosine LR + Knowledge Distillation | Better convergence |
| 🥉 Lower | Coordinate Attention | +0.3-0.5% mAP |
| 🥉 Lower | Mosaic augmentation | Better generalization |

---

## 10. Reference Implementations

- **[YOLOv8/v11](https://github.com/ultralytics/ultralytics)** — TAL, DFL, decoupled head
- **[RTMDet](https://github.com/open-mmlab/mmdetection)** — TAL, efficient design
- **[RTMPose](https://github.com/open-mmlab/mmpose)** — SimCC for keypoints
- **[SCRFD](https://github.com/deepinsight/insightface)** — Current SOTA for face detection

---

## License

MIT License

## Acknowledgments

- Original YuNet: [libfacedetection](https://github.com/ShiqiYu/libfacedetection)
- Training framework: [libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train)
