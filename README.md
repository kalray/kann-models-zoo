# Kalray Neural Network Models
<img width="25%" src="./utils/materials/kalray_logo.png"></a></br>
![ACE5.0.0](https://img.shields.io/badge/Coolidge2-ACE5.0.0-g)
![Classifiers](https://img.shields.io/badge/Classifiers-28-blue)
![Object-Detect](https://img.shields.io/badge/Object%20detection-27-blue)
![Segmentation](https://img.shields.io/badge/Segmentation-07-blue)</br>

The repository purposes neural networks models __ready to compile & run__ on MPPA®.</br>
This is complementary to KaNN™ SDK for model generation and enhance __AI solutions__.

# Requirements
### Hardware requirements
Host machine(s):
  - x86, x86_64 or ARM (aarch64)
  - DDR RAM 8.0Go min
  - HDD disk 16 Go min
  - PCIe Gen 4

Supported acceleration card(s):
  - ![A](https://img.shields.io/badge/Coolidge2-K300-g)
  - ![A](https://img.shields.io/badge/Coolidge2-Turbocard4-g)

### Software requirements
Ubuntu 22.04LTS: ![A](https://img.shields.io/badge/Coolidge2-ACE%205.0.0-g) ![Python3.10.2](https://img.shields.io/badge/Python3.10.2-orange)

# Contents
Neural networks are divided into 3 types:
- [classifiers](./networks/classifiers/README.md)
- [object-detection](./networks/object-detection/README.md)
- [segmentation](./networks/segmentation/README.md)

| Classification (Regnet-x-1.6g)                  | Object-detection (Yolov8)                        | Segmentation (Deeplabv3+)                      |
|-------------------------------------------------|--------------------------------------------------|------------------------------------------------|
| <img src="./utils/materials/cat_class.jpg"></a> | <img src="./utils/materials/cat_detect.jpg"></a> | <img src="./utils/materials/cat_segm.jpg"></a> |

*images has been realized with the model of this repository and KaNN™ SDK solution 

## List of Neural Networks
All networks are proposed into selected Neural Network architectures, such as:

#### __Classifiers__ : complete list can be found here: [link](./networks/classifiers/README.md)
* DenseNet
* EfficientNet
* Inception
* ResNet
* RegNet
* MobileNet
* NasNet
* SqueezeNet
* VGG

#### __Object-detection__ : complete list can be found here: [link](./networks/object-detection/README.md)
* EfficientDet
* Faster-RCNN
* FCN
* RetinatNet
* SSD
* YOLO

#### __Segmentation__ : complete list can be found here: [link](./networks/segmentation/README.md)
* DeeplabV3+
* Mask-RCNN
* UNet2D
* YOLOv8


## How models are packaged ?
Each model is packaged to be compiled and run for KaNN SDK. It is one DIRectory, where you could find:
- a pre-processing python script: `input_preparator.py`
- a post-processing directory: `output_preparator/`
- a model dir: with model file (*.pb, *.onnx, *.tflite) depending of its implementation
- configuration files (*.yaml) for generation:
    * network.yaml :           batch 1 - FP32 - nominal performance
    * network_fp16.yaml :      batch 1 - FP16 - nominal performance
    * network_best.yaml : FP16/INT8* - best performance (including batch, fit memory alignment)
    NB: INT8 is for *.tflite model only
- sources : model reference paper (arxiv, iccv, ..), open-source repository (github, ...)
- license : associated to the model proposed
