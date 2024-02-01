# Kalray Neural Network Models

<img width="25%" src="./utils/materials/kalray_logo.png"></a></br>
![ACE5.0.0](https://img.shields.io/badge/Coolidge2-ACE5.0.0-g)
![ACE5.0.0](https://img.shields.io/badge/Coolidge1-ACE4.13.0-g) \
![Classifiers](https://img.shields.io/badge/Classifiers-29-blue)
![Object-Detect](https://img.shields.io/badge/Object%20detection-26-blue)
![Segmentation](https://img.shields.io/badge/Segmentation-08-blue)</br>

The repository purposes neural networks models __ready to compile & run__ on MPPA®.</br>
This is complementary to KaNN SDK for model generation and enhance __AI solutions__.


## Pre-requisites: configure the SW environment

Please source the Kalray Neural Network™ python environment:</br>
``` $ source $HOME/.local/share/kann/venv*/bin/activate ```


Hints:
* If the python environment does not exist, please use the installation command:</br>
``` kann-install ``` </br>
* If the Kalray's AccessCore® environment does not exist, please source it at:</br>
``` $ source /opt/kalray/accesscore/kalray.sh ```</br>

## Contents

Models are divided into 3 types:
- [classifiers](./networks/classifiers/README.md)
- [objet-detection](./networks/objet-detection/README.md)
- [segmentation](./networks/segmentation/README.md)

How models are packaged ? it is one DIRectory, where you could find:
- a pre-processing python script: `input_preparator.py`
- a post-processing directory: `output_preparator/`
- a model dir: with model file (*.pb, *.onnx, *.tflite) depending of its implementation
- configuration files (*.yaml) for generation:
    * network.yaml :           batch 1 - FP32 - nominal performance
    * network_fp16.yaml :      batch 1 - FP16 - nominal performance
    * network_best.yaml : FP16/INT8* - best performance (including batch, fit memory alignment)
    NB: INT8 is for *.tflite model only
- sources : model reference paper (arxiv, iccv, ..), open-source repository (github, ...)
- license : dedicated to model proposed 


## Generated IR model for inference on MPPA®

Use the following command to generate an model to run on the MPPA®:
```bash
$ ./generate <configuration_file.yaml> -d <generated_path_dir>
...
```
it will provide you into the path directory `generated_path_dir`: 
* a serialized binary file (network contents with runtime and context information)
* a network.dump.yaml file (a copy of the configuration file used)
* a log file of the generation

Please refer to Kalray documentation and KaNN user manual provided for more details

## Run inference from IR model
Use the following command to start quickly the inference of the IR model just generated:
```bash
$ ./run_demo mppa <generated_path_dir> ./utils/sources/street_0.jpg
```

To disable the L2 cache at runtime:
```bash
$ L2=0 ./run_demo mppa <generated_path_dir> ./utils/sources/street_0.jpg
```

To disable the display:
```bash
$ ./run_demo mppa <generated_path_dir> ./utils/sources/street_0.jpg --no-display
```

To disable the replay:
```bash
$ ./run_demo mppa <generated_path_dir> ./utils/sources/street_0.jpg --no-replay
```

To show help, use:
```bash
$ ./run_demo --help or ./run_demo help
```

To run on the CPU target (in order to compare results):
```bash
$ ./run_demo cpu <configuration_file.yaml> ./utils/sources/street_0.jpg --verbose
```

## List of Neural Networks provided for pre-compilation

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


## Custom Layers for extended neural networks

According the Kalray's documentation in KaNN manual, users have the possibility to integrate \
custom layers in case of layer are not supported by KaNN. So, follow those steps (more details in KaNN user manual):
1. Implement the python function callback to ensure that KaNN generator is able to support the Layer
2. Imlement the layer python class to ensure that arguments are matching with the C function
3. Implement C function into the SimpleMapping macro, provided in the example.
4. Build C function with Kalray makefile and reuse it for inference

To ensure to use all extended neural networks provided in the repository, such as YOLOv8 for example \
the DIR `kann_custom_layers` contents the support of :
 * SeLU (KaNN example)
 * Split
 * Slice
 * SiLU
 * Mish

Please find the few steps to use it, for example YOLOv8:

0. Patch the kann-generator wheel into the python environment
```bash
$ source /opt/kalray/accesscore/kalray.sh
(kvxtools)
$ source $HOME/.local/share/kann/venv/bin/activate
(venv)(kvxtools)
$ ./kann_custom_layers/patch-kann-generator.sh
```

1. Buid custom kernels:
```bash
$ make -BC kann_custom_layers O=$PWD/output
```

2. Generate model:
```bash
$ PYTHONPATH=$PWD/kann_custom_layers ./generate $PWD/networks/object-detection/yolov8n/onnx/network_best.yaml -d yolov8n
```

3. Run demo with generated DIR and new kernels compiled (.pocl file) for the MPPA(R)
```bash
$ L2=0 KANN_POCL_DIR=$PWD/output/opencl_kernels/ ./run_demo mppa yolov8n ./utils/sources/street/street_6.jpg
```


## Requirements
### Hardware requirements
Host machine(s):
  - x86, x86_64 or ARM (aarch64)
  - DDR RAM 8.0Go min
  - HDD disk 16 Go min
  - PCIe Gen3 or 4

Acceleration card(s):
  - ![A](https://img.shields.io/badge/Coolidge1-K200-g)
  - ![A](https://img.shields.io/badge/Coolidge2-K300-g)
  - ![A](https://img.shields.io/badge/Coolidge2-Turbocard4-g)


#### OS:
* Ubuntu 18.04LTS: ![A](https://img.shields.io/badge/Coolidge1-ACE%204.13.0-g) ![Python3.6.9](https://img.shields.io/badge/Python3.6.9-orange)
* Ubuntu 22.04LTS: ![A](https://img.shields.io/badge/Coolidge2-ACE%205.0.0-g) ![Python3.10.2](https://img.shields.io/badge/Python3.10.2-orange)
