## pohsuanh 16-825 Assignment 2: Single View to 3D



## Table of Contents
0. [Setup](#0-setup)
1. [Exploring Loss Functions](#1-exploring-loss-functions)
2. [Reconstructing 3D from single view](#2-reconstructing-3d-from-single-view)
3. [Exploring other architectures / datasets](#3-exploring-other-architectures--datasets-choose-at-least-one-more-than-one-is-extra-credit)
## 0. Setup

## 1. Exploring loss functions
This section will involve defining a loss function, for fitting voxels, point clouds and meshes.

### 1.1. Fitting a voxel grid (5 points)


### 1.2. Fitting a point cloud (5 points)


### 1.3. Fitting a mesh (5 points)


## 2. Reconstructing 3D from single view


### 2.1. Image to voxel grid (20 points)
```
python train_model.py --type 'vox' --num_workers 16 --batch_size 16
```

Note that I reduce the batch_size from default 32 to 16 due to the long training time (It will take 1 day to train a single run if not reducing the batch size). I reduce the batch size to 16 due to accelarate the training time.

Evaluation command :
```
python eval_model.py --type 'vox' --load_checkpoint --vis_freq 100
```


### 2.2. Image to point cloud (20 points)



### 2.3. Image to mesh (20 points)
python train_model.py --type 'mesh' --num_workers 12 --batch_size 16


### 2.4. Quantitative comparisions(10 points)


### 2.5. Analyse effects of hyperparams variations (10 points)


### 2.6. Interpret your model (15 points)
shape correspondences from AltasNet

## 3. Exploring other architectures / datasets. (Choose at least one! More than one is extra credit)

### 3.1 Implicit network (10 points)


### 3.2 Parametric network (10 points)


### 3.3 Extended dataset for training (10 points)
