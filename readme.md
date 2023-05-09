# Pool Contest by Computer Vision Zone

In this repository, the video "Shot-Predictor-Video.mp4" from the pool shot predictor competition was processed and it was tried to predict whether the billiard ball could score. The competition has expired. However, the project continues for educational purposes.

The labeling process was done with the RoboFlow annotation tool.
Billiard balls, cue stick and table holes were determined as classes to be detected. Initially, the TensorFlow object detection API was used to detect these classes. But later because of low precision, the YOLOv8 model was used.

TFOD model: ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
YOLOv8 detection model: yolov8x.pt
YOLOv8 segmentation model: yolov8x-seg.pt
