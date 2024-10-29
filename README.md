# Kalray Neural Network Models

<img width="25%" src="./utils/materials/kalray_logo.png"></a></br>

![ACE5.3.0](https://img.shields.io/badge/Coolidge2-ACE--5.3.0-g)
![ACE5.3.1](https://img.shields.io/badge/Coolidge2-ACE--5.3.1-g)
![Classifiers](https://img.shields.io/badge/Classifiers-29-blue)
![Object-Detect](https://img.shields.io/badge/Object%20detection-28-blue)
![Segmentation](https://img.shields.io/badge/Segmentation-8-blue)</br>

The KaNN™ Model Zoo repository provides a list of neural networks models __ready to compile & run__ on MPPA®
manycore processor. This comes on top of KaNN™ tool for model generation and enhance __AI solutions__ onto Kalray
processor.

## SDK Kalray Neural Network (KaNN)

Kalray Neural Network (KaNN) is a SDK included in AccessCore Embedded (ACE) compute offer to optimize AI inference
on our dedicated processor called MPPA® (last generation, the 3rd, is named Coolidge 2). It is composed by:

* __generator__ : a python wheel to parse, optimize and paralellize an intermediate representation of a neural
  networks. Thanks to the runtime, it gives you then the opportunity to run the algorithm directly on the MPPA®
* __runtime__ : optimized libraries (in ASM/C/C++) to execute each operation nodes.

ACE-5.3.0 / ACE-5.3.1 (KaNN) supports: Tensorflow, TFlite, ONNX and Pytorch/ONNX.

## Contents

To quickly deploy a neural network on the MPPA®, a WIKI note is available [here](WIKI.md):
* [Kalray neural networks (KaNN) framework description](./WIKI.md#kalray-neural-networks-kann-framework-description)
* [Pre-requisites: SW environment \& configuration](./WIKI.md#pre-requisites-sw-environment--configuration)
* [How models are packaged](./WIKI.md#how-models-are-packaged)
* [Generate a model to run on the processor (MPPA®)](./WIKI.md#generate-a-model-to-run-on-the-processor-mppa)
* [Evaluate the neural network inference on the MPPA®](./WIKI.md#evaluate-the-neural-network-inference-on-the-mppa)
* [Run neural network as a demo](./WIKI.md#run-neural-network-as-a-demo)
* [Custom Layers for extended neural networks](./WIKI.md#custom-layers-for-extended-neural-networks)
* [Jupyter Notebooks](./WIKI.md#jupyter-notebooks)

CNN Models are divided into 3 types of Machine Vision applications:
* [classification](./networks/classifiers/README.md)
* [object-detection](./networks/object-detection/README.md)
* [segmentation](./networks/segmentation/README.md)

The examples below illustrates the kind of predictions you must have:

| Classification (Regnet-x-1.6g)                                           | Object-detection (Yolov8n)                                                | Segmentation (Deeplabv3+)                                               |
|--------------------------------------------------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------|
| <img height="240" width="240" src="./utils/materials/cat_class.jpg"></a> | <img height="240" width="240" src="./utils/materials/cat_detect.jpg"></a> | <img height="240" width="240" src="./utils/materials/cat_segm.jpg"></a> |

*images has been realized with the model of this repository and KaNN™ SDK solution 

## List of Neural Networks

All networks are proposed into selected Neural Network architectures, such as:

__Classifiers__ : complete list can be found [here](./networks/classifiers/README.md)

* DenseNet
* EfficientNet
* Inception
* ResNet
* RegNet
* MobileNet
* NasNet
* SqueezeNet
* VGG

__Object-detection__ : complete list can be found [here](./networks/object-detection/README.md)

* EfficientDet
* Faster-RCNN
* FCN
* RetinatNet
* SSD
* YOLO

__Segmentation__ : complete list can be found [here](./networks/segmentation/README.md)

* DeeplabV3+
* Mask-RCNN
* UNet
* YOLO


## Requirements

### Hardware requirements
Host machine(s):
* x86_64 CPU
* DDR RAM 8Go min
* HDD disk 32 Go min
* PCIe Gen3 min, Gen4 recommended

Acceleration card(s):
* ![A](https://img.shields.io/badge/Coolidge2-K300-g)
* ![A](https://img.shields.io/badge/Coolidge2-Turbocard4-g)

### Software requirements
* ![U22](https://img.shields.io/badge/Ubuntu-22.04%20LTS-orange)
  ![Ker](https://img.shields.io/badge/Linux%20Kernel-5.15.0-red)
* ![ACE](https://img.shields.io/badge/Coolidge2-ACE--5.3.0-g)
* ![ACE](https://img.shields.io/badge/Coolidge2-ACE--5.3.1-g)
* ![Python](https://img.shields.io/badge/Python-3.10-blue)
  ![Python](https://img.shields.io/badge/Python-3.11-blue)
