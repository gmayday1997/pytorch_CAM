# Discriminative Localization with Image-Level Annotation
Pytorch implementation of "Learning Deep Features for Discriminative Localization"

B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba
Learning Deep Features for Discriminative Localization.
Computer Vision and Pattern Recognition (CVPR), 2016.
[[PDF](http://arxiv.org/pdf/1512.04150.pdf)][[Project Page](http://cnnlocalization.csail.mit.edu/)]

### Contents
1. [Basic installation](#basic-installation)
2. [Data and Pretrain Model](#data-and-pretrain-model)
3. [Demo](#demo)
4. [Beyond the demo:Training and Testing](#Beyond-the-demo-training-and-testing)
5. [Results of Action40 Dataset](#Results-of-Action-40-Dataset)

### Basic installation
 Requirements for 'pytorch'(see: [pytorch installation instuctions](http://pytorch.org/))
### Data and Pretrain Model

- Action40 datasets
     - [action_dataset.tar](https://drive.google.com/file/d/0B71WibNFGUgaYkZNR2FqQ0hNOXc/view?usp=sharing)
     - [action_dataset.tar](https://drive.google.com/file/d/0B71WibNFGUgaYkZNR2FqQ0hNOXc/view?usp=sharing)
     - [class_id.json](https://drive.google.com/file/d/0B71WibNFGUgackc5NW1QQ0JiOFk/view?usp=sharing)
- Pretrain Models
     - [action_dataset.tar](https://drive.google.com/file/d/0B71WibNFGUgaYkZNR2FqQ0hNOXc/view?usp=sharing)
     - [vgg_16_converted_from_caffe.pth](https://drive.google.com/file/d/0B71WibNFGUgad1dWeS1lbHV3R0E/view?usp=sharing)
     - [action_40_pretrain_model.pth](https://drive.google.com/file/d/0B71WibNFGUgaOXg5YzRMRXFPRlU/view?usp=sharing)
  
### Demo
After sucessfully completing [Basic installation](#installation) and [Data and Pretrain Model](#data-and-pretrain-model), you will be 
ready to run the demo.
```Shell
cd $CAM_ROOT
./demo.py
```
### Beyond the demo: Traing and Testing
**Train**
set train_flag = True in action40_config.py
run train_action40.py
**Test**
set train_flag = False in action40_config.py
run test_action40.py

### Results of Action40 Dataset
![alt tag](https://github.com/gmayday1997/pytorch-CAM/blob/master/results/cam_0.jpg)
![alt tag](https://github.com/gmayday1997/pytorch-CAM/blob/master/results/cam_67.jpg)
![alt tag](https://github.com/gmayday1997/pytorch-CAM/blob/master/results/cam_311.jpg)
![alt tag](https://github.com/gmayday1997/pytorch-CAM/blob/master/results/cam_400.jpg)
![alt tag](https://github.com/gmayday1997/pytorch-CAM/blob/master/results/cam_644.jpg)
![alt tag](https://github.com/gmayday1997/pytorch-CAM/blob/master/results/cam_760.jpg)
![alt tag](https://github.com/gmayday1997/pytorch-CAM/blob/master/results/cam_851.jpg)
