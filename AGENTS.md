# YuNet Face Detector Training

Training program for YuNet face detector. The model predicts:
- Facial bounding boxes
- Five facial keypoints (left eye-center, right eye-center, nose, left lip corner, right lip corner)

## Goals

1. **Primary**: Reproduce author's results [DONE]
2. **Secondary**: Obtain better results than author's [UNDONE]
3. **Learning**: Understand face detection neural networks - architectures, components, training, caveats, issues, difficulties [TBD, going bad, it's hard]

## Project Structure

```
source/
├── models/          # YuNet architecture (backbone, neck, head, blocks)
├── dataset/         # WiderFace dataset handling and transforms
├── losses/          # Loss functions (EIoU, MTL)
├── training.py      # Training loop
├── simota.py        # SimOTA label assignment
├── targets.py       # Target generation
└── postprocessing.py # NMS and inference post-processing
```

## Key Technical Details

- **Architecture**: YuNet is a lightweight face detector based on YOLO-style design
- **Dataset**: WiderFace (convert with `convert_widerface.py`)
- **Label Assignment**: SimOTA (dynamic label assignment)
- **Losses**: Multi-task learning with EIoU for box regression

## AI Assistant Guidelines

When helping with this project:
- Explain concepts thoroughly - this is a learning project
- Reference the author's repo when comparing implementations
- Prioritize code clarity over micro-optimizations
- When suggesting architecture changes, explain the tradeoffs;
- If you don't know true answer just say it. Don't imagine, or warn me that you are unsure and starting to imagine;
- Don't be too gentle. If you see any shit in my code just say it. It's absolutely no hard-feelings and heart-breaking.

## Useful Links

- https://github.com/ShiqiYu/libfacedetection.train - author's repository
- https://github.com/WongKinYiu?tab=repositories - useful repos for "true" YOLO architectures
