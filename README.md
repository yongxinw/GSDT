# GSDT
### Joint Object Detection and Multi-Object Tracking with Graph Neural Networks
This is the official PyTorch implementation of our paper: "[**Joint Object Detection and Multi-Object Tracking with Graph Neural Networks**](https://arxiv.org/abs/2006.13164)". Our project website and video demos are [here](http://www.xinshuoweng.com/projects/GNNDetTrk/). If you find our work useful, we'd appreciate you citing our paper as follows:

```
@article{Wang2020_GSDT, 
author = {Wang, Yongxin and Kitani, Kris and Weng, Xinshuo}, 
journal = {arXiv:2006.13164}, 
title = {{Joint Object Detection and Multi-Object Tracking with Graph Neural Networks}}, 
year = {2020} 
}
```

<p>
<img align="center" width="48%" src="https://github.com/yongxinw/GSDT/blob/main/main1.gif">
<img align="center" width="48%" src="https://github.com/yongxinw/GSDT/blob/main/main2.gif">
</p>
<p>
<img align="center" width="48%" src="https://github.com/yongxinw/GSDT/blob/main/main3.gif">
<img align="center" width="48%" src="https://github.com/yongxinw/GSDT/blob/main/main4.gif">
</p>

## Introduction
Object detection and data association are critical components in multi-object tracking (MOT) systems. Despite the fact that the two components are dependent on each other, prior work often designs detection and data association modules separately which are trained with different objectives. As a result, we cannot back-propagate the gradients and optimize the entire MOT system, which leads to sub-optimal performance. To address this issue, recent work simultaneously optimizes detection and data association modules under a joint MOT framework, which has shown improved performance in both modules. In this work, we propose a new instance of joint MOT approach based on Graph Neural Networks (GNNs). The key idea is that GNNs can model relations between variable-sized objects in both the spatial and temporal domains, which is essential for learning discriminative features for detection and data association. Through extensive experiments on the MOT15/16/17/20 datasets, we demonstrate the effectiveness of our GNN-based joint MOT approach and show the state-of-the-art performance for both detection and MOT tasks.

## Usage
### Dependencies
We recommend using [**anaconda**](https://www.anaconda.com/) for managing dependency and environments. You may follow the commands below to setup your environment. 
```angular2
conda create -n dev python=3.6
conda activate dev
pip install -r requirements.txt
```

We use the [**PyTorch Geometric**](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) package for the implementation of our Graph Neural Network based architecture.
```angular2
bash install_pyg.sh <CUDA_version>  # we used CUDA_version=cu101 
``` 

Build Deformable Convolutional Networks V2 (DCNv2)
```angular2
cd ./src/lib/models/networks/DCNv2
bash make.sh
``` 

To automatically generate output tracking as videos, please install `ffmpeg`
```angular2
conda install ffmpeg=4.2.2
```

### Data preperation
We follow the same dataset setup as in [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT). Please refer to their [DATA ZOO](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) for data download and preperation.  

To prepare [2DMOT15](https://motchallenge.net/data/2D_MOT_2015/) and [MOT20](https://motchallenge.net/data/MOT20/) data, you can directly download from the [**MOT Challenge**](https://motchallenge.net/) website, and format each directory as follows:
```
MOT15
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
MOT20
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
```
Then change the seq_root and label_root in src/gen_labels_15.py and src/gen_labels_20.py accordingly, and run:
```
cd src
python gen_labels_15.py
python gen_labels_20.py
```
This will generate the desired label format of 2DMOT15 and MOT20. The seqinfo.ini files are required for 2DMOT15 and can be found here [[Google]](https://drive.google.com/open?id=1kJYySZy7wyETH4fKMzgJrYUrTfxKlN1w), [[Baidu],code:8o0w](https://pan.baidu.com/s/1zb5tBW7-YTzWOXpd9IzS0g).

## Inference
Download and save the pretrained weights for each dataset by following the links below:

| Dataset    |  Model |
|------------|--------|
|2DMOT15     | [**model_mot15.pth**](https://drive.google.com/file/d/1K_6yN1jD7fpmGN23Z4NgsROO5mZRf6ay/view?usp=sharing) |
|MOT17       | [**model_mot17.pth**](https://drive.google.com/file/d/1Aj6h3UCgFgw69ffxh-OqodvaJhQ9PS4m/view?usp=sharing) |
|MOT20       | [**model_mot20.pth**](https://drive.google.com/file/d/1cX92Sp9NpWmL-UwAQJyR88AjGp6UpOgW/view?usp=sharing) |

Run one of the following command to reproduce our paper's tracking performance on the MOT Challenge.
```angular2
cd ./experiments
track_gnn_mot_AGNNConv_RoIAlign_mot15.sh <path/to/model_mot15>
track_gnn_mot_AGNNConv_RoIAlign_mot17.sh <path/to/model_mot17>
track_gnn_mot_AGNNConv_RoIAlign_mot20.sh <path/to/model_mot20>
``` 

To clarify, currently we directly used the MOT17 results as MOT16 results for submission. That is, our MOT16 and MOT17 results and models are identical.
## Training
We are currently in the process of cleaning the training code. We'll release as soon as we can. Stay tuned!

# Performance on MOT Challenge
You can refer to [MOTChallenge website](https://motchallenge.net/results/MOT20/?det=All) for performance of our method. For your convenience, we summarize results below:
| Dataset    |  MOTA | IDF1 | MT | ML | IDS |
|--------------|-----------|--------|-------|----------|----------|
|2DMOT15  | 60.7 | 64.6 |  47.0% | 10.5% | 477 |
|MOT16       | 66.7 | 69.2 | 38.6% | 19.0% | 959 |
|MOT17       | 66.2 | 68.7 | 40.8% | 18.3% | 3318 |
|MOT20       | 67.1 | 67.5 | 53.1% | 13.2% | 3133 |

## Acknowledgement
A large part of the code is borrowed from [FairMOT](https://github.com/ifzhang/FairMOT). We appreciate their great work!
