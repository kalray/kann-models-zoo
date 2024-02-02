<img width="20%" src="../../utils/materials/kalray_logo.png"></a>

## List of Object detection neural networks
Object detection neural networks are provided into selected architectures:
* EfficientDet
* Faster-RCNN
* RetinatNet
* SSD
* YOLO

Please find the complete list here:

| NAME                                              |  FLOPs  | Params | Implementation | Dataset  | Input Size | ACE status                                              |
|:--------------------------------------------------|:-------:|:------:|:---------------|:--------:|:----------:|:--------------------------------------------------------|
| [RetinaNet-resnet50](./retinanet-resnet50)        | 122.4 G | 37.9 M | onnx           | COCO2017 |  512x512   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [RetinaNet-resnet50](./retinanet-resnet50-mlperf) | 299.6 G | 37.9 M | onnx           | COCO2017 |  800x800   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [RetinaNet-resnet101](./retinanet-resnet101)      | 161.4 G | 56.9 M | onnx           | COCO2017 |  512x512   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [SSD-mobileNetV1](./ssd-mobilenet-v1-mlperf)      | 2.45 G  | 6.7 M  | tensorflow1    | COCO2017 |  300x300   | ![Fail](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [SSD-mobileNetV2](./ssd-mobilenet-v2)             | 3.71 G  | 16.1 M | tensorflow1    | COCO2017 |  300x300   | ![Fail](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [SSD-resnet34](./ssd-resnet34-mlperf)             | 433.1 G | 20.0 M | tensorflow1    | COCO2017 | 1200x1200  | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [YOLOv3](./yolov3)                                | 65.99 G | 61.9 M | tensorflow1    | COCO2017 |  416x416   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [YOLOv3](./yolov3)                                | 66.12 G | 61.9 M | onnx           | COCO2017 |  416x416   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [YOLOv3-Tiny](./yolov3-tiny)                      | 5.58 G  | 8.9 M  | tensorflow1    | COCO2017 |  416x416   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [YOLOv4](./yolov4)                                | 66.17G  | 64.3 M | onnx           | COCO2017 |  416x416   | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red)   |
| [YOLOv4-Tiny](./yolov4-tiny)                      | 6.92 G  | 6.1 M  | onnx           | COCO2017 |  416x416   | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow)  |
| [YOLOv4-CSP-Mish](./yolov4-csp-mish)              | 114.7 G | 52.9 M | onnx           | COCO2017 |  608x608   | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow)  |
| [YOLOv4-CSP-Relu](./yolov4-csp-relu)              | 109.2 G | 52.9 M | onnx           | COCO2017 |  608x608   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [YOLOv4-CSP-S-Mish](./yolov4-csp-s-mish)          | 21.64 G | 8.3 M  | onnx           | COCO2017 |  608x608   | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow)  |
| [YOLOv4-CSP-S-Relu](./yolov4-csp-s-relu)          | 19.16 G | 8.3 M  | onnx           | COCO2017 |  608x608   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [YOLOv5s](./yolov5s)                              | 17.3 G  | 7.2 M  | onnx           | COCO2017 |  640x640   | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow)  |
| [YOLOv5m6-TI-Lite](./yolov5m6-relu)               | 52.45 G | 35.5 M | onnx           | COCO2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [YOLOv5s6-TI-Lite](./yolov5s6-relu)               | 17.44 G | 12.6 M | onnx           | COCO2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [YOLOv7](./yolov7)                                | 107.8 G | 36.9 M | onnx           | COCO2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [YOLOv7-Tiny](./yolov7-tiny)                      | 13.7 G  | 6.2 M  | onnx           | COCO2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)     |
| [YOLOv8n](./yolov8n)                              |  8.7 G  | 3.2 M  | onnx           | COCO2017 |  640x640   | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow)  |
| [YOLOv8s](./yolov8s)                              | 28.6 G  | 11.2 M | onnx           | COCO2017 |  640x640   | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow)  |
| [YOLOv8m](./yolov8m)                              | 78.9 G  | 25.9 M | onnx           | COCO2017 |  640x640   | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow)  |

![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow) means that you have to use Custom layer \
feature to run the model (and would be integrated in nextACE release). \
Please see HOW_TO.md to use it.  