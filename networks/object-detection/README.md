<img width="20%" src="../../utils/materials/kalray_logo.png"></a>

## List of Object detection neural networks
Object detection neural networks are provided into selected architectures:
* EfficientDet
* Faster-RCNN
* RetinatNet
* SSD
* YOLO

Please find the complete list here: </br>
![ModelsPass](https://img.shields.io/badge/ACE5.0-15/26-blue)

<!-- START AUTOMATED TABLE -->
| NAME                                              |  FLOPs  | Params | mAP-50 | Implementation | Dataset  | Input Size | ACE status                                            | Batch | Coolidge2⁽¹⁾<br/>Performance (FPS) |
|:--------------------------------------------------|:-------:|:------:|:------:|:---------------|:--------:|:----------:|:------------------------------------------------------|:-----:|:----------------------------------:|
| [EfficientDet-D0](./efficientdet-d0)              | 10.2 G  | 3.9 M  | 53.0 % | tensorflow1    | COCO2017 |  512x512   | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |       |                                    |
| [EfficientDet-D0](./efficientdet-d0)              | 10.2 G  | 3.9 M  | 53.0 % | onnx           | COCO2017 |  512x512   | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |       |                                    |
| [FasterRCNN-resnet50](./faster-rcnn-rn50)         | 94.1 G  | 26.8 M | 35.0 % | onnx           | COCO2017 |  512x512   | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |       |                                    |
| [RetinaNet-resnet50](./retinanet-resnet50)        | 122.4 G | 37.9 M |  TBC   | onnx           | COCO2017 |  512x512   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |   1   |                27.6                |
| [RetinaNet-resnet50](./retinanet-resnet50-mlperf) | 299.6 G | 37.9 M | 35.8 % | onnx           | COCO2017 |  800x800   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |   1   |                27.6                |
| [RetinaNet-resnet101](./retinanet-resnet101)      | 161.4 G | 56.9 M |  TBC   | onnx           | COCO2017 |  512x512   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |   1   |                23.8                |
| [SSD-mobileNetV1](./ssd-mobilenet-v1-mlperf)      | 2.45 G  | 6.7 M  |  TBC   | tensorflow1    | COCO2017 |  300x300   | ![Fail](https://img.shields.io/badge/ACE5.0-pass-g)   |   5   |               452.8                |
| [SSD-mobileNetV2](./ssd-mobilenet-v2)             | 3.71 G  | 16.1 M |  TBC   | tensorflow1    | COCO2017 |  300x300   | ![Fail](https://img.shields.io/badge/ACE5.0-pass-g)   |   5   |               329.3                |
| [SSD-resnet34](./ssd-resnet34-mlperf)             | 433.1 G | 20.0 M |  TBC   | tensorflow1    | COCO2017 | 1200x1200  | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |   1   |                66.0                |
| [YOLOv3](./yolov3)                                | 65.99 G | 61.9 M | 55.3 % | tensorflow1    | COCO2017 |  416x416   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |   5   |                88.7                |
| [YOLOv3](./yolov3)                                | 66.12 G | 61.9 M | 55.3 % | onnx           | COCO2017 |  416x416   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |   5   |                90.2                |
| [YOLOv3-Tiny](./yolov3-tiny)                      | 5.58 G  | 8.9 M  | 33.1 % | tensorflow1    | COCO2017 |  416x416   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |   3   |               656.8                |
| [YOLOv4](./yolov4)                                | 66.17G  | 64.3 M | 57.3 % | onnx           | COCO2017 |  416x416   | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |       |                                    |
| [YOLOv4-Tiny](./yolov4-tiny)                      | 6.92 G  | 6.1 M  | 40.2 % | onnx           | COCO2017 |  416x416   | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |   3   |               575.8                |
| [YOLOv4-CSP-Mish](./yolov4-csp-mish)              | 114.7 G | 52.9 M | 55.0 % | onnx           | COCO2017 |  608x608   | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |   3   |                9.9                 |
| [YOLOv4-CSP-Relu](./yolov4-csp-relu)              | 109.2 G | 52.9 M | 54.0 % | onnx           | COCO2017 |  608x608   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |   3   |                44.0                |
| [YOLOv4-CSP-S-Mish](./yolov4-csp-s-mish)          | 21.64 G | 8.3 M  | 44.6 % | onnx           | COCO2017 |  608x608   | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |   3   |                25.7                |
| [YOLOv4-CSP-S-Relu](./yolov4-csp-s-relu)          | 19.16 G | 8.3 M  | 42.6 % | onnx           | COCO2017 |  608x608   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |   3   |               119.9                |
| [YOLOv5s](./yolov5s)                              | 17.3 G  | 7.2 M  | 56.8 % | onnx           | COCO2017 |  640x640   | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |   3   |               181.4                |
| [YOLOv5m6-TI-Lite](./yolov5m6-relu)               | 52.45 G | 35.5 M | 62.9 % | onnx           | COCO2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |   2   |               125.5                |
| [YOLOv5s6-TI-Lite](./yolov5s6-relu)               | 17.44 G | 12.6 M | 56.0 % | onnx           | COCO2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |   2   |               300.1                |
| [YOLOv7](./yolov7)                                | 107.8 G | 36.9 M | 69.7 % | onnx           | COCO2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |   1   |                31.2                |
| [YOLOv7-Tiny](./yolov7-tiny)                      | 13.7 G  | 6.2 M  |   nc   | onnx           | COCO2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |   3   |               208.7                |
| [YOLOv8n](./yolov8n)                              |  8.7 G  | 3.2 M  | 44.9 % | onnx           | COCO2017 |  640x640   | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |   3   |               337.4                |
| [YOLOv8s](./yolov8s)                              | 28.6 G  | 11.2 M | 58.8 % | onnx           | COCO2017 |  640x640   | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |   2   |               146.7                |
| [YOLOv8m](./yolov8m)                              | 78.9 G  | 25.9 M | 61.3 % | onnx           | COCO2017 |  640x640   | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |   1   |                75.0                |
<!-- END AUTOMATED TABLE -->

[1] FP16 datatype inference at best batch/performance, in Frame per seconds (FPS)\
Host machine: AMD Ryzen 5 12-Core - DRAM 32.0 GB | Acceleration card: K300