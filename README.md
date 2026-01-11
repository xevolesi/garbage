# YuNet PyTorch

A clean PyTorch reimplementation of [YuNet](https://link.springer.com/article/10.1007/s11633-023-1423-y) — a lightweight face detection model with facial landmark prediction.

Author's repos:
- [Training](https://github.com/ShiqiYu/libfacedetection.train);
- [Inference](https://github.com/ShiqiYu/libfacedetection).

# Why?

I don't like `mmdetection`. From my point of view it's to overcomplicated. I prefer clean `PyTorch` code.

# Installation

I used `Windows 11` with `python3.11`, `CUDA-13.1`. So, get `python3.11` and corresponding `venv` first. Than do the following:

```code
python3.11 -m venv .venv
source .venv/bin/activate or .venv\Scripts\Activate.ps1 if you are on Windows
pip install -r requirements.txt
```

Since there are no super frameworks like `mmdetection` or `mmcv` here, this repo should be easy to install. For specific version of python packages please check `requirements.txt`.


# Notes and differencies

## Augmentations

I use letter boxing during preprocessing for validation and inference. You can see more in `source/dataset/transforms`.

## Dataset

I use custom dataset format. It consists of images and single CSV file. CSV file must have the following columns:
```code
image,boxes,key_points,subset
```
where
- `image` is the **ABSOLUTE** path to the image;
- `boxes` is the list of `[x1, y1, x2, y2]` boxes. Each column contains single list with multiple boxes, e.g. N faces on image = N boxes in list;
- `key_points` is the list of facial key points in `[x, y, visibility]` format. Each column contains single list of lists of points, e.g. N faces on image = N boxes = N key points sets = N lists of lists with length 5 points (left eye, right eye, nose, left lip corner, right lib corner), each point is the list of 3 values `[x, y, visibility]`. `visibility` is the `visibility weight` that is `1` if this point in visible and `-1` otherwise. Note, that, during parsing (you can check `convert_widerface.py` script) `-1` visibility weight will be replaces to `0`. All questions about `labelv2` you can ask authors directly, since i don't know the rationale behind such visibility weight logic. As far as i remember guys from [insightface](https://github.com/deepinsight/insightface) use `labelv2` too; 
- `subset` is the string with the subset name (`train` or `val`);
- `source` is the name of the dataset.

To translate your WIDERFACE dataset to this custom format you need to perform the following steps:
1. Download [WIDERFACE](http://shuoyang1213.me/WIDERFACE/);
2. Unzip downloaded images in some directoryl
2. Download [labelv2 folder](https://github.com/ShiqiYu/libfacedetection.train/tree/master/data/widerface/labelv2) and place it inside WIDERFACE folder.
3. Run `convert_widerface.py` to create `.csv` file that will be saved into WIDERFACE folder.

Conversion procedure is fairly simple, so i strongly encourage you to read the script and understand how the labels are parsed.

Why i use such dataset format:
- This is just a habit I acquired while participating in Kaggle competitions;
- In my opinion, .csv is the most convenient format for storing labels in a single file.

That's basically all, nothing special.

Example:
```code
# Let 'D:\datasets\widerface' be the target folder;
# Unzip train, val, test subsets of WIDERFACE into 'D:\datasets\widerface';
# Place labelv2 folder into 'D:\datasets\widerface'.
# Target structure:

D:\datasets\widerface\:
    | - labelv2
    | - WIDER_test
    | - WIDER_train
    | - WIDER_val
    | - widerface_main.csv # This .csv will be added by `convert_widerface.py` script
```

To train on your custom dataset you need to convert your dataset into the format described above. Just be sure that `image` column in CSV file consists of **ABSOLUTE** paths to your images.

## Metrics

During training pipeline logs `mAP` values on evaluation set. Note, that this is **NOT** `mAP` from WIDERFACE `eval_tools`. This is `mAP` metric calculated using `torchmetrics` and `faster-coco-eval`, so it's basically 3 values: `mAP small`, `mAP medium`, `mAP large`. These values are logged during training just to show training progress. More about this values you can read in `torchmetrics` documentation. Final WIDERFACE evaluation metrics you can compute using `evaluate_widerface.py`.

# Results

Author's results were taken from their training [repo](https://github.com/ShiqiYu/libfacedetection.train).

Mine reults were calculated using `evaluate_widerface.py` script using original image size, `score_thresh=0.02` and `nms_thresh=0.45`.

| Model | Easy | Medium | Hard | Params |
|-------|------|--------|------|--------|
| YuNet-n (original) | 89.2 | 88.3 | 81.1 | ~75K |
| YuNet-n (this repo) | 89.4 | 88.3 | 81.1 | ~75K |
