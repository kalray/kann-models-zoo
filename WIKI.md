# Kalray Neural Network Models

<img width="50%" src="./utils/materials/mppa-processor.jpg"></a></br>

![ACE5.1.0](https://img.shields.io/badge/Coolidge2-ACE5.1.0-g)
![Classifiers](https://img.shields.io/badge/Classifiers-28-blue)
![Object-Detect](https://img.shields.io/badge/Object%20detection-28-blue)
![Segmentation](https://img.shields.io/badge/Segmentation-09-blue)</br>

The repository purposes neural networks models __ready to compile & run__ on MPPA®.</br>
This is complementary to KaNN™ SDK for model generation and enhance __AI solutions__.

## Table of contents
<!-- TOC -->
  * [Kalray neural networks (KaNN) description](#kalray-neural-networks-kann-description)
  * [Pre-requisites: configure the SW environment](#pre-requisites-configure-the-sw-environment)
  * [How models are packaged](#how-models-are-packaged)
  * [Generate a model to run on MPPA®](#generate-a-model-to-run-on-mppa)
  * [Run inference from kann-video-demo](#run-inference-from-kann-video-demo)
  * [Custom Layers for extended neural networks](#custom-layers-for-extended-neural-networks)
<!-- TOC -->

## Kalray neural networks (KaNN) description

<img width="500" src="./utils/materials/CNN.png"></a></br>

KaNN is a Kalray software purpose, available in the SDK AccessCore Embedded (ACE) offer. It leverages the possibility \
to parse and analyze a Convolution Neural Network (figure above) from different SW environments \
such as ONNX, Tensorflow, TFlite, PyTorch; and generate a MPPA code to achieve the best performance efficiency. \
This repository does not contain any information about the use of hte API, but it helps to deploy AI solutions. \
For details, please do not hesitate to read the documentation 😏 or contact us directly.

So, to deploy your solution from an identified neural networks, the steps are all easy 😃 :
1. From a CNN, generate a model (no HW dependencies)
2. Run model from demo application (python + cpp host application, included in the repository and ACE software)

## Pre-requisites: configure the SW environment

Please source the Kalray Neural Network™ python environment:</br>
``` $ source $HOME/.local/share/kann/venv*/bin/activate ```

If it does not exist, please configure a specific virtual python environment:
```bash
export KANN_ENV=$HOME/.local/share/python3-kann-venv
python3 -m venv $KANN_ENV
```

Source your python environment:
```bash 
source $KANN_ENV/bin/activate
```

Install local KaNN wheel and its dependencies (it supposed that release is install in $HOME):
```bash
pip install $HOME/ACE5.2.0/KaNN-generator/kann-5.2.0-py3*.whl
```
Finally, the python requirements of the repo:
```bash 
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
Please see kalray lounge install procedure detailed at:
[link](https://lounge.kalrayinc.com/hc/en-us/articles/14613836001308-ACE-5-2-0-Content-installation-release-note-and-Getting-Started-Coolidge-v2)


## How models are packaged

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


## Generate a model to run on MPPA®

Use the following command to generate an model to run on the MPPA®:
```bash
$ ./generate <configuration_file.yaml> -d <generated_path_dir>
...
```
It will provide you into the path directory `generated_path_dir`: 
* a serialized binary file (network contents with runtime and context information)
* a network.dump.yaml file (a copy of the configuration file used)
* a log file of the generation

Please refer to Kalray documentation and KaNN user manual provided for more details !

## Run inference from kann-video-demo
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
1. Patch the kann-generator wheel into the python environment
```bash
$ source /opt/kalray/accesscore/kalray.sh
(kvxtools)
$ source $HOME/.local/share/kann/venv-*/bin/activate
(venv)(kvxtools)
$ ./kann_custom_layers/patch-kann.sh
```

2. Then, buid custom kernels for MPPA:
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
