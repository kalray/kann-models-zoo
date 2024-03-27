# Kalray Neural Network Models

<img width="25%" src="./utils/materials/kalray_logo.png"></a></br>
![ACE5.1.0](https://img.shields.io/badge/Coolidge2-ACE5.1.0-g)
![Classifiers](https://img.shields.io/badge/Classifiers-27-blue)
![Object-Detect](https://img.shields.io/badge/Object%20detection-25-blue)
![Segmentation](https://img.shields.io/badge/Segmentation-10-blue)</br>

The KaNN™ Model Zoo repository provides a list of neural networks models __ready to compile & run__ on Kalray MPPA® manycore processor.</br>
This comes on top of KaNN™ tool for model generation and enhance __AI solutions__ onto Kalray processor.


## Contents

Models are divided into 3 types:
- [classifiers](./networks/classifiers/README.md)
- [object-detection](./networks/object-detection/README.md)
- [segmentation](./networks/segmentation/README.md)

| Classification (Regnet-x-1.6g)                               | Object-detection (Yolov8n)                                    | Segmentation (Deeplabv3+)                                  |
|--------------------------------------------------------------|---------------------------------------------------------------|------------------------------------------------------------|
| <img height="240" src="./utils/materials/cat_class.jpg"></a> | <img height="240" src="./utils/materials/cat_detect.jpg"></a> | <img height="240" src="./utils/materials/cat_segm.jpg"></a> |

*images has been realized using model from this repository and KaNN™ SDK solution

Please refer to [WIKI.md](./WIKI.md) to :
+ Run quickly a Neural Network on the MPPA® 
+ Deploy your AI solution

## List of Neural Networks 

All networks have been categorized based on Neural Network architectures:

#### __Classifiers__ : complete list can be found [here](./networks/classifiers/README.md):
* DenseNet
* EfficientNet
* Inception
* MobileNet
* NasNet
* ResNet
* RegNet
* SqueezeNet
* VGG

#### __Object-detection__ : complete list can be found [here](./networks/object-detection/README.md):
* EfficientDet
* RetinatNet
* SSD
* YOLO

#### __Segmentation__ : complete list can be found [here](./networks/segmentation/README.md):
* DeeplabV3+
* Fully Convolution Network (FCN)
* U-Net
* YOLO


## Requirements
### Hardware requirements
  - x86, x86_64 or ARM CPU processor
  - 8GB DDR (minimum)
  - 16GB HDD disk (minimum)
  - PCIe interface (16-lane Gen4 recommended)

### Software requirements
* ![U22](https://img.shields.io/badge/Ubuntu-22.04%20LTS-orange)
* ![ACE](https://img.shields.io/badge/Coolidge2-ACE%205.1.0-g)
* ![Python3.10.2](https://img.shields.io/badge/Python-3.10.2-blue)

### Supported Kalray acceleration card(s)
  - ![A](https://img.shields.io/badge/Coolidge2-K300-g)
  - ![A](https://img.shields.io/badge/Coolidge2-Turbocard4-g)

