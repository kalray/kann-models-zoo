<img width="30%" src="https://upload.wikimedia.org/wikipedia/commons/4/46/Logo-KALRAY.png"></a>

## List of Classifiers Neural networks
Classifiers neural networks are provided into several architectures:
* DenseNet
* EfficientNet
* Inception
* MobileNet
* NasNet
* Resnet
* RegNet
* SqueezeNet
* VGG

All models have been trained on :
**ImageNet Large Scale Visual Recognition Challenge 2012 ([ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/))** dataset</br>
Except for:
* R-CNN ([ILSVRC2013](https://www.image-net.org/challenges/LSVRC/2013/))

Please find the complete list here: </br>
![ModelsPass](https://img.shields.io/badge/ACE5.0-20/29-blue)

<!-- START AUTOMATED TABLE -->
| NAME                                         | FLOPs    | Params   | accTop1 | accTop5 | Implem.     | ACE status                                            | Batch-size | Coolidge2⁽¹⁾<br/>Performance (FPS) |
|:---------------------------------------------|:---------|:---------|:--------|:--------|:------------|:------------------------------------------------------|:----------:|:----------------------------------:|
| [alexNet](./alexnet)                         | 1.335 G  | 60.9 M   | 56.52 % | 79.06 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-fail-red) |     5      |               505.3                |
| [denseNet-121](./densenet-121)               | 5.718 G  | 8.04 M   | 74.43 % | 91.97 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     3      |               301.8                |
| [denseNet-169](./densenet-169)               | 6.777 G  | 14.27 M  | 75.6 %  | 92.81 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     2      |               223.7                |
| [efficientNet-B0](./efficientnet-b0)         | 1.004 G  | 5.26 M   | 77.69 % | 93.53 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-fail-red) |     5      |               191.9                |
| [efficientNet-B4](./efficientnet-b4)         | 11.727 G | 16.83 M  | 83.38 % | 96.59 % | onnx        | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |     1      |                7.7                 |
| [efficientNetLite-B4](./efficientnetlite-b4) | 2.785 G  | 12.96 M  | 80.4 %  | -       | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-fail-red) |     5      |               228.3                |
| [googleNet](./googlenet)                     | 3.014 G  | 6.61 M   | 69.8 %  | 89.6 %  | tensorflow1 | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |              1,781.5               |
| [googleNet](./googlenet)                     | 3.014 G  | 6.62 M   | 69.8 %  | 89.5 %  | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |              1,898.4               |
| [inception-resnetv2](./inception-resnetv2)   | 13.27 G  | 55.9 M   | 80.3 %  | 95.3 %  | tensorflow1 | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |               250.9                |
| [inception-V3](./inception-v3)               | 11.42 G  | 27.16 M  | 77.2 %  | 93.4 %  | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-fail-red) |     5      |                35.6                |
| [mobileNet-V1](./mobilenet-v1)               | 1.124 G  | 4.16 M   | 70.9 %  | 89.9 %  | tensorflow1 | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |              1,003.2               |
| [mobileNet-V2](./mobilenet-v2)               | 0.893 G  | 3.54 M   | 71.88 % | 90.29 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     4      |               716.5                |
| [mobileNet-V3-large](./mobilenet-v3-large)   | 0.465 G  | 5.47 M   | 74.04 % | 91.34 % | onnx        | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |            |                                    |
| [nasnet](./nasnet)                           | 0.650 G  | 4.36 M   | 73.45 % | 91.51 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-fail-red) |     5      |               592.4                |
| [rcnn](./rcnn)                               | 1.444 G  | 57.69 M  | -       | -       | onnx        | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |            |                                    |
| [regNet-x-1.6g](./regNet-x-1.6g)             | 3.240 G  | 9.17 M   | 77.04 % | 93.44 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |              1,221.3               |
| [regNet-x-8.0g](./regNet-x-8.0g)             | 16.052 G | 39.53 M  | 79.34 % | 94.68 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |               443.4                |
| [resnet18](./resnet18)                       | 3.642 G  | 11.70 M  | 69.75 % | 89.07 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |              1,610.5               |
| [resnet34](./resnet34)                       | 7.348 G  | 21.81 M  | 73.31 % | 91.42 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |               926.0                |
| [resnet50](./resnet50)                       | 7.770 G  | 25.63 M  | 74.93 % | 92.38 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |               606.2                |
| [resnet50v1.5](./resnet50v1.5)               | 8.234 G  | 25.53 M  | 76.13 % | 92.86 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |               595.3                |
| [resnet50v2](./resnet50v2)                   | 8.209 G  | 25.5 M   | 75.81 % | 92.82 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-fail-red) |     5      |               462.0                |
| [resnext50](./resnext50)                     | 8.436 G  | 25.0 M   | 77.62 % | 93.69 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |               435.4                |
| [resnet101](./resnet101)                     | 15.221 G | 44.70 M  | 77.37 % | 93.54 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |               360.1                |
| [resnet152](./resnet152)                     | 22.680 G | 60.4 M   | 78.31 % | 94.04 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |               261.4                |
| [squeezeNet](./squeezenet)                   | 0.714 G  | 1.23 M   | 58.17 % | 80.62 % | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |              2,885.2               |
| [vgg-16](./vgg-16)                           | 31.006 G | 138.36 M | 71.3 %  | 90.1 %  | tensorflow1 | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     4      |               151.5                |
| [vgg-19](./vgg-19)                           | 37.683 G | 12.85 M  | 71.3 %  | 90.0 %  | tensorflow1 | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     4      |               132.6                |
| [xception](./xception)                       | 9.07 G   | 22.9 M   | 79.0 %  | 94.5 %  | tensorflow1 | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     5      |               181.9                |
<!-- END AUTOMATED TABLE -->

[1] FP16 datatype inference at best batch/performance, in Frame per seconds (FPS)\
Host machine: AMD Ryzen 5 12-Core - DRAM 32.0 GB | Acceleration card: K300
