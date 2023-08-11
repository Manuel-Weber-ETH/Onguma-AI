# Welcome to Onguma AI
Website: www.Onguma-AI.com
Contact: OngumaAI@gmail.com
Developer: Manuel Weber, Onguma Nature Reserve

This repository contains all the code necessary to customize the algorithm to your camera trap images.

![Onguma Black Rhino](https://github.com/Manuel-Weber-ETH/Onguma-AI/assets/118481837/20e1a941-2acf-427b-a809-bf70d4cbaf5a)

The repository contains 3 scripts and 1 pre-trained model:
## training_2class_250k.py
training.py trains a convolutional neural network (CNN) on a set of training images belonging to two classes, and saves it.
## app_version_1.5.py
app.py turns the CNN into a software with graphical user interface (GUI). This script can be turned into an .exe file, which can then be transferred to other computers.
## rhino_detection_model.h5
rhino_detection_model.h5 is the output of training_2class_250k.py, a CNN capable of detecting rhinos in camera trap images, trained on 15000 high-quality images.
## rhino_detection_model_250k_8_8.h5
rhino_detection_model.h5 is the output of training_2class_250k.py, a CNN capable of detecting rhinos in camera trap images, trained on 250000 low-quality images.

# Policy and copyright
The scripts may be used freely for non-commercial, conservation-linked projects.
