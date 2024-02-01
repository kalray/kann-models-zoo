<img width="20%" src="../../utils/materials/kalray_logo.png"></a>

## List of Segmentation neural networks
Object detection neural networks are provided into selected architectures:
* DeeplabV3+
* EfficientDet
* Fully Convolution Network (FCN)
* Mask-RCNN
* UNet
* YOLO

Please find the complete list here: </br>
![ModelsPass](https://img.shields.io/badge/ACE5.0-8/13-blue)

| NAME                                                      | FLOPs   | Params |  mIoU  |  Input  | Dataset      | Implementation | ACE status                                            | Batch-size | Coolidge2⁽¹⁾<br/>Performance (FPS) |
|:----------------------------------------------------------|:--------|-------:|:------:|:-------:|:-------------|:---------------|:------------------------------------------------------|:----------:|:----------------------------------:|
| [DeeplabV3Plus-mobilenet-V2](./deeplabv3plus-mobilenetv2) | 17.4 G  |  2.0 M |   -    | 512x512 | VOC-COCO2017 | tensorflow1    | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     1      |                32.0                |
| [DeeplabV3Plus-mobilenet-V3](./deeplabv3plus-mobilenetv3) | 10.4 G  | 42.3 M | 60.3 % | 512x512 | VOC-COCO2017 | onnx           | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |            |                                    |
| [DeeplabV3Plus-Resnet50](./deeplabv3plus-resnet50)        | 65.3 G  | 12.0 M |   -    | 416x416 | COCO2014     | tensorflow1    | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     1      |                25.5                |
| [DeeplabV3Plus-Resnet50](./deeplabv3plus-resnet50)        | 21.6 G  |    ? M |   -    | 416x416 | VOC-COCO2017 | onnx           | ![Fail](https://img.shields.io/badge/ACE5.0-fail-red) |            |                                    |
| EfficientDet-D0                                           |         |        |        |         |              |                |                                                       |            |                                    |
| [FCN-Resnet50](./fcn_resnet50)                            | 276.9 G | 32.9 M | 60.5 % | 512x512 | VOC-COCO2017 | onnx           | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     1      |                10.4                |
| [FCN-Resnet101](./fcn_resnet101)                          | 432.2 G | 51.8 M | 63.7 % | 512x512 | VOC-COCO2017 | onnx           | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     1      |                13.5                |
| MaskRCNN-Resnet50                                         |         |        |        |         |              |                |                                                       |            |                                    |
| [UNet-2D-medical](./unet2d-tiny-med)                      | 24.4 G  |  7.7 M |   -    | 256x256 | MRI-BRAIN    | onnx           | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     1      |               347.4                |
| [UNet-2D-indus](./unet2d-tiny-ind)                        | 36.7 G  | 1.85 M |   -    | 512x512 | DAGM-2007    | tensorflow1    | ![Pass](https://img.shields.io/badge/ACE5.0-pass-g)   |     1      |                85.7                |
| UNet-3D                                                   |         |        |        |         |              |                |                                                       |            |                                    |
| YOLOv7-seg                                                |         |        |        |         |              |                |                                                       |            |                                    |
| YOLOv8s-seg                                               |         |        |        |         |              |                |                                                       |            |                                    |

[1] FP16 datatype inference at best batch/performance, in Frame per seconds (FPS)\
Host machine: AMD Ryzen 5 12-Core - DRAM 32.0 GB | Acceleration card: K300