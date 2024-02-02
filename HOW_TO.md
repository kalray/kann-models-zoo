# Kalray Neural Network Models

<img width="25%" src="./utils/materials/kalray_logo.png"></a></br>
![ACE5.0.0](https://img.shields.io/badge/Coolidge2-ACE5.0.0-g)
![Classifiers](https://img.shields.io/badge/Classifiers-28-blue)
![Object-Detect](https://img.shields.io/badge/Object%20detection-27-blue)
![Segmentation](https://img.shields.io/badge/Segmentation-07-blue)</br>

The repository purposes neural networks models __ready to compile & run__ on MPPA®.</br>
This is complementary to KaNN™ SDK for model generation and enhance __AI solutions__.

# HOW TO

## Configure the SW environment

Please source the Kalray Neural Network™ python environment:</br>
``` $ source $HOME/.local/share/kann/venv*/bin/activate ```

Hints:
* If the python environment does not exist, please use the installation command:</br>
``` kann-install ``` </br>
* If the Kalray's AccessCore® environment does not exist, please source it at:</br>
``` $ source /opt/kalray/accesscore/kalray.sh ```</br>

## Generate IR model for inference on MPPA®

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

## Use custom Layers for extended neural networks

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
$ source $HOME/.local/share/kann/venv/bin/activate
(venv)(kvxtools)
$ ./kann_custom_layers/patch-kann-generator.sh
```

2. Buid custom kernels:
```bash
$ make -BC kann_custom_layers O=$PWD/output
```

3. Generate model:
```bash
$ PYTHONPATH=$PWD/kann_custom_layers ./generate $PWD/networks/object-detection/yolov8n/onnx/network_best.yaml -d yolov8n
```

4. Run demo with generated DIR and new kernels compiled (.pocl file) for the MPPA(R)
```bash
$ L2=0 KANN_POCL_DIR=$PWD/output/opencl_kernels/ ./run_demo mppa yolov8n ./utils/sources/street/street_6.jpg
```
