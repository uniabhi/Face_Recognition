
# InsightFace: 2D and 3D Face Analysis Project

By Jia Guo and [Jiankang Deng](https://jiankangdeng.github.io/)

## License

The code of InsightFace is released under the MIT License.

## ArcFace Video Demo

[![ArcFace Demo](https://github.com/deepinsight/insightface/blob/master/resources/facerecognitionfromvideo.PNG)](https://www.youtube.com/watch?v=y-D1tReryGA&t=81s)

Please click the image to watch the Youtube video. For Bilibili users, click [here](https://www.bilibili.com/video/av38041494?from=search&seid=11501833604850032313).

## Recent Update

**`2019.04.14`**: We will launch a Light-weight Face Recognition challenge/workshop on ICCV 2019.

**`2019.04.04`**: Arcface achieved state-of-the-art performance (5/109) on the NIST Face Recognition Vendor Test (FRVT) (1:1 verification)
[report](https://www.nist.gov/sites/default/files/documents/2019/04/04/frvt_report_2019_04_04.pdf) (name: Imperial-000). Our solution is based on [MS1MV2+DeepGlintAsian, ResNet100, ArcFace loss]. 

**`2019.02.08`**: Please check [https://github.com/deepinsight/insightface/tree/master/recognition](https://github.com/deepinsight/insightface/tree/master/recognition) for our parallel training code which can easily and efficiently support one million identities on a single machine (8* 1080ti).

**`2018.12.13`**: Inference acceleration [TVM-Benchmark](https://github.com/deepinsight/insightface/wiki/TVM-Benchmark).

**`2018.10.28`**: Light-weight attribute model [Gender-Age](https://github.com/deepinsight/insightface/tree/master/gender-age). About 1MB, 10ms on single CPU core. Gender accuracy 96% on validation set and 4.1 age MAE.

**`2018.10.16`**: We achieved state-of-the-art performance on [Trillionpairs](http://trillionpairs.deepglint.com/results) (name: nttstar) and [IQIYI_VID](http://challenge.ai.iqiyi.com/detail?raceId=5afc36639689443e8f815f9e) (name: WitcheR). 

## Contents
[Deep Face Recognition](#deep-face-recognition)
- [Introduction](#introduction)
- [Training Data](#training-data)
- [Train](#train)
- [Pretrained Models](#pretrained-models)
- [Verification Results On Combined Margin](#verification-results-on-combined-margin)
- [Test on MegaFace](#test-on-megaface)
- [512-D Feature Embedding](#512-d-feature-embedding)
- [Third-party Re-implementation](#third-party-re-implementation)

[Face Alignment](#face-alignment)

[Face Detection](#face-detection)

[Citation](#citation)

[Contact](#contact)

## Deep Face Recognition

### Introduction

In this repository, we provide training data, network settings and loss designs for deep face recognition.
The training data includes the normalised MS1M, VGG2 and CASIA-Webface datasets, which were already packed in MXNet binary format.
The network backbones include ResNet, MobilefaceNet, MobileNet, InceptionResNet_v2, DenseNet, DPN.
The loss functions include Softmax, SphereFace, CosineFace, ArcFace and Triplet (Euclidean/Angular) Loss.


![margin penalty for target logit](https://github.com/deepinsight/insightface/raw/master/resources/arcface.png)

Our method, ArcFace, was initially described in an [arXiv technical report](https://arxiv.org/abs/1801.07698). By using this repository, you can simply achieve LFW 99.80%+ and Megaface 98%+ by a single model. This repository can help researcher/engineer to develop deep face recognition algorithms quickly by only two steps: download the binary dataset and run the training script.

### Training Data

All face images are aligned by [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html) and cropped to 112x112:

Please check [Dataset-Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) for detail information and dataset downloading.


* Please check *src/data/face2rec2.py* on how to build a binary face dataset. Any public available *MTCNN* can be used to align the faces, and the performance should not change. We will improve the face normalisation step by full pose alignment methods recently.

### Train

1. Install `MXNet` with GPU support (Python 2.7).

```
pip install mxnet-cu90
```

2. Clone the InsightFace repository. We call the directory insightface as *`INSIGHTFACE_ROOT`*.

```
git clone --recursive https://github.com/deepinsight/insightface.git
```

3. Download the training set (`MS1M-Arcface`) and place it in *`$INSIGHTFACE_ROOT/datasets/`*. Each training dataset includes at least following 6 files:

```Shell
    faces_emore/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
```

The first three files are the training dataset while the last three files are verification sets.

4. Train deep face recognition models.
In this part, we assume you are in the directory *`$INSIGHTFACE_ROOT/recognition/`*.
```Shell
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
```

Place and edit config file:
```Shell
cp sample_config.py config.py
vim config.py # edit dataset path etc..
```

We give some examples below. Our experiments were conducted on the Tesla P40 GPU.

(1). Train ArcFace with LResNet100E-IR.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r100 --loss arcface --dataset emore
```
It will output verification results of *LFW*, *CFP-FP* and *AgeDB-30* every 2000 batches. You can check all options in *config.py*.
This model can achieve *LFW 99.80+* and *MegaFace 98.3%+*.

(2). Train CosineFace with LResNet50E-IR.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r50 --loss cosface --dataset emore
```

(3). Train Softmax with LMobileNet-GAP.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network m1 --loss softmax --dataset emore
```

(4). Fine-turn the above Softmax model with Triplet loss.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network m1 --loss triplet --lr 0.005 --pretrained ./models/m1-softmax-emore,1
```



