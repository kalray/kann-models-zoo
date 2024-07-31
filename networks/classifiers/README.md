<img width="30%" src="https://upload.wikimedia.org/wikipedia/commons/4/46/Logo-KALRAY.png"></a>

## List of Classifiers Neural networks
This repository gives access to following classifiers Neural Networks main architecture:
* DenseNet
* EfficientNet
* Inception
* MobileNet
* NasNet
* Resnet
* RegNet
* SqueezeNet
* VGG

Please find the complete list here: </br>
![ModelsPass](https://img.shields.io/badge/ACE5.3-28%20pass-g)
![Models](https://img.shields.io/badge/Total%20model-29-blue)

<!-- START AUTOMATED TABLE -->
| NAME                                         | FLOPs    | Params   | accTop1 | accTop5 | Framework   | Dataset     | ACE status                                             | GPU⁽⁰⁾  | Coolidge2⁽¹⁾ | Eff (%) |
|:---------------------------------------------|:---------|:---------|:--------|:--------|:------------|-------------|:-------------------------------------------------------|:-------:|:------------:|:-------:|
| [alexNet](./alexnet)                         | 1.335 G  | 60.9 M   | 56.52 % | 79.06 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    | 41,580  |     905      |  5.7 %  | 
| [denseNet-121](./densenet-121)               | 5.718 G  | 8.04 M   | 74.43 % | 91.97 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |  6,535  |     356      |  9.0 %  |
| [denseNet-169](./densenet-169)               | 6.777 G  | 14.27 M  | 75.6 %  | 92.81 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |  4,925  |     263      |  7.9 %  |
| [efficientNet-B0](./efficientnet-b0)         | 1.004 G  | 5.26 M   | 77.69 % | 93.53 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    | 16,000  |    236⁽²⁾    |  1.0 %  |
| [efficientNet-B4](./efficientnet-b4)         | 11.727 G | 16.83 M  | 83.38 % | 96.59 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |  1,855  |    30⁽²⁾     |  1.6 %  |
| [efficientNetLite-B4](./efficientnetlite-b4) | 2.785 G  | 12.96 M  | 80.4 %  | -       | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    | 10,345  |     443      |  5.0 %  |
| [googleNet](./googlenet)                     | 3.014 G  | 6.61 M   | 69.8 %  | 89.6 %  | tensorflow1 | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |         |    1,960     | 25.9 %  |
| [googleNet](./googlenet)                     | 3.014 G  | 6.62 M   | 69.8 %  | 89.5 %  | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    | 30,585  |    1,910     | 25.5 %  |
| [inception-resnetv2](./inception-resnetv2)   | 13.27 G  | 55.9 M   | 80.3 %  | 95.3 %  | tensorflow1 | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |         |     266      | 15.6 %  |
| [inception-V3](./inception-v3)               | 11.42 G  | 27.16 M  | 77.2 %  | 93.4 %  | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |  9,820  |      38      |  2.0 %  |
| [mobileNet-V1](./mobilenet-v1)               | 1.124 G  | 4.16 M   | 70.9 %  | 89.9 %  | tensorflow1 | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |         |    1,416     |  7.1 %  |
| [mobileNet-V2](./mobilenet-v2)               | 0.893 G  | 3.54 M   | 71.88 % | 90.29 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    | 37,855  |    1,235     |  3.1 %  |
| [mobileNet-V3-large](./mobilenet-v3-large)   | 0.465 G  | 5.47 M   | 74.04 % | 91.34 % | onnx        | ILSVRC 2012 | ![Ext](https://img.shields.io/badge/ACE5.3-ext-orange) | 34,195  |    640⁽³⁾    |  1.1 %  |
| [nasnet](./nasnet)                           | 0.650 G  | 4.36 M   | 73.45 % | 91.51 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    | 28,945  |    1,217     |  3.1 %  |
| [regNet-x-1.6g](./regnet-x-1.6g)             | 3.240 G  | 9.17 M   | 77.04 % | 93.44 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    | 15,400  |    1,220     | 17.4 %  |
| [regNet-x-8.0g](./regnet-x-8.0g)             | 16.052 G | 39.53 M  | 79.34 % | 94.68 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |  8,540  |     513      | 36.5 %  |
| [resnet18](./resnet18)                       | 3.642 G  | 11.70 M  | 69.75 % | 89.07 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    | 38,570  |    1,625     | 26.2 %  |
| [resnet34](./resnet34)                       | 7.348 G  | 21.81 M  | 73.31 % | 91.42 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    | 22,620  |     953      | 31.0 %  |
| [resnet50](./resnet50)                       | 7.770 G  | 25.63 M  | 74.93 % | 92.38 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    | 16,980  |     620      | 21.2 %  |
| [resnet50v1.5](./resnet50v1.5/onnx)          | 8.234 G  | 25.53 M  | 76.13 % | 92.86 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    | 15,340  |     619      | 22.5 %  |
| [resnet50v1.5](./resnet50v1.5/tflite)        | 8.234 G  | 25.53 M  | - %     | - %     | tflite      | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |         |     875      | 16.0 %  |
| [resnet50v2](./resnet50v2)                   | 8.209 G  | 25.5 M   | 75.81 % | 92.82 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |         |     595      | 21.7 %  |
| [resnext50](./resnext50)                     | 8.436 G  | 25.0 M   | 77.62 % | 93.69 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    | 12,445  |     455      | 17.1 %  |
| [resnet101](./resnet101)                     | 15.221 G | 44.70 M  | 77.37 % | 93.54 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |  9,180  |     387      | 26.8 %  |
| [resnet152](./resnet152)                     | 22.680 G | 60.4 M   | 78.31 % | 94.04 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |  6,485  |     276      | 27.7 %  |
| [squeezeNet](./squeezenet)                   | 0.714 G  | 1.23 M   | 58.17 % | 80.62 % | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    | 41,555  |    3,126     |  9.9 %  |
| [vgg-16](./vgg-16)                           | 31.006 G | 138.36 M | 71.3 %  | 90.1 %  | tensorflow1 | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |         |     151      | 20.7 %  |
| [vgg-19](./vgg-19)                           | 37.683 G | 12.85 M  | 71.3 %  | 90.0 %  | tensorflow1 | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |         |     131      | 22.9 %  |
| [xception](./xception)                       | 9.07 G   | 22.9 M   | 79.0 %  | 94.5 %  | tensorflow1 | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.3-pass-g)    |         |     476      | 19.2 %  |
<!-- END AUTOMATED TABLE -->

![Ext](https://img.shields.io/badge/ACE5.3-ext-yellow) means that you have to use Custom layer
feature to run the model (and would be integrated in next ACE release).

Please see WIKI.md to use it.

Nota Bene:
```
* Host machine: AMD Ryzen 5 12-Core - DRAM 32.0 GB
* tensorflow1:  tensorflow 1.x (frozen model) <= 1.15
* tensorflow2:  tensorflow 2.x (saved model) <= 2.11
* tflite:       tensorflow lite <= 2.11
* onnx:         onnx <= 1.13
```

