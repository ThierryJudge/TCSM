## Transformation Consistent Self-ensembling Model for Semi-supervised Medical Image Segmentation

Pytorch implementation of TCSM <br/>

## Paper
[Transformation Consistent Self-ensembling Model for Semi-supervised Medical Image Segmentation](https://arxiv.org/abs/1911.01376)
<br/>
<p align="center">
  <img src="figure/framework.png">
</p>

## Installation
* Install Pytorch 1.1.0 and CUDA 9.0
* Clone this repo
```
git clone https://github.com/xmengli999/TCSM
cd TCSM
```

## Data Preparation
* Download [Skin dataset](https://challenge.kitware.com/#phase/5841916ccad3a51cc66c8db0), [REFUGE dataset](https://refuge.grand-challenge.org/REFUGE2018/), [Liver CT dataset](https://competitions.codalab.org/competitions/17094#phases) <br/>
* Put the data under `./data/`

## Train
* cd `messidor_scripts` and specify the pretrain model in `train_fold.sh`
* Run `sh train_fold.sh` to start the training process

## Evaluate
* Specify the model path in `eval_fold.sh`
* Run `sh eval_fold.sh` to start the evaluation.

## Acknowledgement
CBAM module is reused from the [Pytorch implementation of CBAM](https://github.com/Jongchan/attention-module).

## Note
* Contact: Xiaomeng Li (xmengli999@gmail.com)
