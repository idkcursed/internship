# Tyre Detection and Counting System

A YOLOv8-based computer vision system for detecting and counting tyres in images and videos [file:11]. The model distinguishes between inner and outer tyre diameters and provides accurate counting with object tracking [file:11].

## Features

- **Dual Detection**: Detects both inner diameter (Inner_dia) and outer diameter (outer_dia) of tyres [file:11]
- **High Accuracy**: Achieves 98.9% mAP@50 and 98.3% mAP@50-95 on validation data [file:11]
- **Image Detection**: Count tyres in static images [file:11]
- **Video Processing**: Track and count tyres entering/exiting a defined line in videos [file:11]
- **Smart Filtering**: Removes nested/overlapping detections using IoU-based filtering [file:11]
- **Object Tracking**: Simple centroid-based tracking for video analysis [file:11]

## Installation


## Dataset

The model is trained on the `tyre` dataset containing [file:11]:
- **Training set**: 212 images
- **Validation set**: 56 images
- **Classes**: 2 (Inner_dia, outer_dia)

## Training

The model was trained for 50 epochs using YOLOv8n (nano) with the following configuration [file:11]:

- **Model**: YOLOv8n
- **Image size**: 640x640
- **Batch size**: 16
- **Epochs**: 50
- **Optimizer**: AdamW (lr=0.001667)
- **Device**: Tesla T4 GPU

### Training Command


## Performance Metrics

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|---------|-----------|
| All | 0.988 | 0.983 | 0.989 | 0.983 |
| Inner_dia | 0.982 | 0.983 | 0.988 | 0.975 |
| outer_dia | 0.994 | 0.982 | 0.991 | 0.991 |

