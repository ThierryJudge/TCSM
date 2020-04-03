## Transformation Consistent Self-ensembling Model for Semi-supervised Medical Image Segmentation

Pytorch implementation of TCSM <br/>

## Paper
[Transformation Consistent Self-ensembling Model for Semi-supervised Medical Image Segmentation](https://arxiv.org/pdf/1903.00348.pdf)
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
* cd `scripts_skin` 
* Run `sh train_50tcsm.sh` to start the training process

## Evaluate
* Specify the model path in `eval.sh`
* Run `sh eval.sh` to start the evaluation.

## Acknowledgement
Some code is reused from the [Pytorch implementation of mean teacher](https://github.com/CuriousAI/mean-teacher). 

## Note
* Contact: Xiaomeng Li (xmengli999@gmail.com)
