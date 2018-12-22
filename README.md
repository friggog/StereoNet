# Pytorch StereoNet
Customized implementation of the [Stereonet guided hierarchical refinement for real-time edge-aware depth prediction](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sameh_Khamis_StereoNet_Guided_Hierarchical_ECCV_2018_paper.pdf)

## Attention: Not accomplished yet
1. Till 2018-12-22, the [https://github.com/meteorshowers/StereoNet](https://github.com/meteorshowers/StereoNet) still not released the code.
And that's the only version of the StereoNet now.
Hope the coder can release the code as soon as possible so I can rectify the codes in my repository. 
2. The approach of computing the cost volume in the StereoNet paper is subtracting the padding image and the other image. Here I changed it to concatenate the two images. If you want to change it to the paper's way, just set it in the train.py when you initialize the net.

## Pre-requirement
+ Pytorch 1.0.0
+ CUDA Toolkit 10
+ numpy

### You can use the anaconda virtual environment to quick start

#### Install Anaconda
```
1. wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
2. bash Anaconda3-5.3.1-Linux-x86_64.sh
```

Please reference to [Ubuntu系统下Anaconda使用方法总结](http://zhixuanli.cn/?p=468) for more information about install conda.

#### Create Virtual Environment according to my environment index 
```conda env create -n your_env_name -f environment.yaml```

## Trainng and Test
#### Switch to the correct python environment
```
conda activate your_env_name
```

#### Start training and test
```python train.py```


 
## Coding Reference
+ [Pyramid Stereo Matching Network (CVPR2018)](https://github.com/JiaRenChang/PSMNet)
+ [gc-net for stereo matching by using pytorch](https://github.com/zyf12389/GC-Net)
+ [An tensorflow raw implementation of paper "End-to-End Learning of Geometry and Context for Deep Stereo Regression"](https://github.com/liuruijin17/GCNet-tensorflow)
