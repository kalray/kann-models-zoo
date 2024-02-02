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

Please find the complete list here:

| NAME                                         | FLOPs    | Params   | Implem.     | ACE status                                             |
|:---------------------------------------------|:---------|:---------|:------------|:-------------------------------------------------------|
| [alexNet](./alexnet)                         | 1.335 G  | 60.9 M   | onnx        | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow) |
| [denseNet-121](./densenet-121)               | 5.718 G  | 8.04 M   | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [denseNet-169](./densenet-169)               | 6.777 G  | 14.27 M  | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [efficientNet-B0](./efficientnet-b0)         | 1.004 G  | 5.26 M   | onnx        | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow) |
| [efficientNet-B4](./efficientnet-b4)         | 11.727 G | 16.83 M  | onnx        | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow) |
| [efficientNetLite-B4](./efficientnetlite-b4) | 2.785 G  | 12.96 M  | onnx        | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow) |
| [googleNet](./googlenet)                     | 3.014 G  | 6.61 M   | tensorflow1 | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [googleNet](./googlenet)                     | 3.014 G  | 6.62 M   | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [inception-resnetv2](./inception-resnetv2)   | 13.27 G  | 55.9 M   | tensorflow1 | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [inception-V3](./inception-v3)               | 11.42 G  | 27.16 M  | onnx        | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow) |
| [mobileNet-V1](./mobilenet-v1)               | 1.124 G  | 4.16 M   | tensorflow1 | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [mobileNet-V2](./mobilenet-v2)               | 0.893 G  | 3.54 M   | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [mobileNet-V3-large](./mobilenet-v3-large)   | 0.465 G  | 5.47 M   | onnx        | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow) |
| [nasnet](./nasnet)                           | 0.650 G  | 4.36 M   | onnx        | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow) |
| [rcnn](./rcnn)                               | 1.444 G  | 57.69 M  | onnx        | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red)  |
| [regNet-x-1.6g](./regNet-x-1.6g)             | 3.240 G  | 9.17 M   | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [regNet-x-8.0g](./regNet-x-8.0g)             | 16.052 G | 39.53 M  | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [resnet18](./resnet18)                       | 3.642 G  | 11.70 M  | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [resnet34](./resnet34)                       | 7.348 G  | 21.81 M  | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [resnet50](./resnet50)                       | 7.770 G  | 25.63 M  | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [resnet50v1.5](./resnet50v1.5)               | 8.234 G  | 25.53 M  | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [resnet50v2](./resnet50v2)                   | 8.209 G  | 25.5 M   | onnx        | ![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow) |
| [resnext50](./resnext50)                     | 8.436 G  | 25.0 M   | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [resnet101](./resnet101)                     | 15.221 G | 44.70 M  | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [resnet152](./resnet152)                     | 22.680 G | 60.4 M   | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [squeezeNet](./squeezenet)                   | 0.714 G  | 1.23 M   | onnx        | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [vgg-16](./vgg-16)                           | 31.006 G | 138.36 M | tensorflow1 | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [vgg-19](./vgg-19)                           | 37.683 G | 12.85 M  | tensorflow1 | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |
| [xception](./xception)                       | 9.07 G   | 22.9 M   | tensorflow1 | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)    |

![Ext](https://img.shields.io/badge/ACE5.0-ext-yellow) means that you have to use Custom layer \
feature to run the model (and would be integrated in nextACE release). \
Please see HOW_TO.md to use it.  