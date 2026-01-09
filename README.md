# YuNet PyTorch

A clean PyTorch reimplementation of [YuNet](https://link.springer.com/article/10.1007/s11633-023-1423-y) — a lightweight face detection model with facial landmark prediction.

Author's repos:
- [Training](https://github.com/ShiqiYu/libfacedetection.train);
- [Inference](https://github.com/ShiqiYu/libfacedetection).

# Why?

I don't like `mmdetection`. From my point of view it's to overcomplicated. I prefer clean `PyTorch` code.

# Notes and differencies

- I use letter boxing during preprocessing for validation and inference. You can see more in `source/dataset/transforms`.

# Results
Author's results were taken from their training [repo](https://github.com/ShiqiYu/libfacedetection.train).

Mine reults were calculated using `evaluate_widerface.py` script using original image size, `score_thresh=0.02` and `nms_thresh=0.45`.

| Model | Easy | Medium | Hard | Params |
|-------|------|--------|------|--------|
| YuNet-n (original) | 89.2 | 88.3 | 81.1 | ~75K |
| YuNet-n (this repo) | 89.4 | 88.3 | 81.1 | ~75K |
