# RankPose: Learning Generalised Feature with Rank Supervision for Head Pose Estimation

## Paper
[RankPose: Learning Generalised Feature with Rank Supervision for Head Pose Estimation](https://arxiv.org/abs/2005.10984)

## Abstract
We address the challenging problem of RGB image-based head pose estimation. We first reformulate head pose representation learning to constrain it to a bounded space. Head pose represented as vector projection or vector angles shows helpful to improving. performance. Further, a ranking loss combined with MSE regression loss is proposed. The ranking loss supervises a neural network with paired samples of the same person and penalises incorrect ordering of pose prediction. Analysis on this new loss function suggests it contributes to a better local feature extractor, where features are generalised to Abstract Landmarks which are pose-related features instead of pose-irrelevant information such as identity, age, and lighting. Extensive experiments show that our method significantly outperforms the current state-of-the-art schemes on public datasets:  
AFLW2000 and BIWI. Our model achieves significant improvements over previous SOTA
MAE on AFLW2000 and BIWI from 4.50 [11] to 3.66 and from 4.0 [24] to 3.71 respectively. 

## Dependencies

+ pytorch >= 0.4.1
+ albumentations
+ opencv2
+ yaml
~~~
pip3 install requirements.txt
~~~

## Datasets

### Train data
[Face Alignment Across Large Poses: A 3D Solution](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)

[300W-LP](https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?usp=sharing)

### Test data
[AFLW2000](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip)
[BIWI Kinect](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)

## Train and test

### Training
~~~
CUDA_VISIBLE_DEVICES=0 python3 train.py ../config/headpose_resnet.yaml
~~~

### Testing
~~~
CUDA_VISIBLE_DEVICES=0 python3 test.py
~~~

### Pretrained model
Will be available for download in the future.

