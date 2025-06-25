# Generalised YOLOX Nano – STM32 model zoo services

Welcome to my adaptation of STM32 model zoo services!

This repository is a stand alone version of STM's ModelZoo services, adapted to support different output amounts and shapes for the YOLOX Nano object detection model. For example, training, evaluation and quantisation are supported if heads are removed or added to the st_yolo_x model. It also supports removing the classification branch from the heads. This project was made for my bachelor thesis, since I wanted to optimise the st_yolo_x model further for person detection using a neuromorphic camera sensor as an input. For examples of models that were trained, quantised and evaluated using this service, you can go to [my other repository](https://github.com/piergatt/YoloOptimiseTools).

## What's new :

Files modified to support more generalised output heads are:
- [cfg_utils.py](common/utils/cfg_utils.py)
- [evaluate.py](object_detection/src/evaluation/evaluate.py)
- [postprocess.py](object_detection/src/postprocessing/postprocess.py)
- [quantize.py](object_detection/src/quantization/quantize.py)
- [train.py](object_detection/src/training/train.py)
- [yolo_loss.py](object_detection/src/training/yolo_loss.py)
- [yolo_x_train_model.py](object_detection/src/training/yolo_x_train_model.py)
- [parse_config.py](object_detection/src/utils/parse_config.py)

This includes a modification to the [yaml](object_detection/user_config.yaml), where you can set the strides of the heads yourself. A full list of my changes can be found in the change logs.

## How to use:

You should mostly follow the README in the original [STModelZoo services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) for help and support with how to use the general program. My solution is meant to be a drag and drop replacement and there should be no need for extra configuration. The only addition is the ability to add the "network_stride" parameter to the post-processing section of the yaml, which is required if you either changed the amount of heads you are using, or changed the output stride of the model.

<div align="center" style="margin-top: 80px; padding: 20px 0;">
    <p align="center">
        <a href="https://www.python.org/downloads/" target="_blank"><img src="https://img.shields.io/badge/python-3.10-blue" /></a>
        <a href="https://www.tensorflow.org/install/pip" target="_blank"><img src="https://img.shields.io/badge/TensorFlow-2.8.4-FF6F00?style=flat&logo=tensorflow&logoColor=#FF6F00&link=https://www.tensorflow.org/install/pip"/></a>
        <a href="https://stedgeai-dc.st.com/home"><img src="https://img.shields.io/badge/STM32Cube.AI-Developer%20Cloud-FFD700?style=flat&logo=stmicroelectronics&logoColor=white"/></a>  
    </p>
</div>

## Disclamer:
You should primarily stick to STM's licensing, however if you comply with those you are free to use my adaptations of STM's code however you wish.