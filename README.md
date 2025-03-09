# YOLOv8 Custom Object Detection

## Dataset
The dataset is in the `project-1-at-2025-03-08-19-32-bd2549ff` directory with the following structure:
- `images/`: ~34 images
- `labels/`: corresponding label files in YOLO format
- `classes.txt`: contains 2 classes - clip and sleeper

## Training

To train the model on the dataset:

```bash
pip install ultralytics
python yolo_train.py
```

This will:
1. Organize the dataset into the YOLO format
2. Train a YOLOv8n model for 20 epochs
3. Save the trained model to `runs/detect/clip_sleeper_model/`

## Inference

To run inference on test images:

```bash
python yolo_predict.py
```

This will:
1. Load the trained model
2. Run predictions on sample test images
3. Save the results with bounding boxes to the `predictions/` folder
# TrackSentry
