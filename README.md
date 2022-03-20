# ENLCA

This project is the official implementation of 'Efficient Non-Local Contrastive Attention for Image Super-Resolution', AAAI22
> **Efficient Non-Local Contrastive Attention for Image Super-Resolution [[Paper](https://arxiv.org/pdf/2201.03794.pdf)] ** 

## Contents
- [ENLCA](#enlca)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Train](#train)
    - [Prepare training data](#prepare-training-data)
    - [Run](#run)
  - [Test](#test)
    - [Quick start](#quick-start)
  - [Results](#results)
    - [Quantitative Results](#quantitative-results)
    - [Visual Results](#visual-results)
  - [Citation](#citation)

## Introduction
Non-Local Attention (NLA) brings significant improvement for Single Image Super-Resolution (SISR) by leveraging intrinsic feature correlation in natural images. However, NLA gives noisy information large weights and consumes quadratic computation resources with respect to the input size, limiting its performance and application. In this paper, we propose a novel Efficient Non-Local Contrastive Attention (ENLCA) to perform long-range visual modeling and leverage more relevant non-local features. Specifically, ENLCA consists of two parts, Efficient Non-Local Attention (ENLA) and Sparse Aggregation. ENLA adopts the kernel method to approximate exponential function and obtains linear computation complexity. For Sparse Aggregation, we multiply inputs by an amplification factor to focus on informative features, yet the variance of approximation increases exponentially. Therefore, contrastive learning is applied to further separate relevant and irrelevant features. To demonstrate the effectiveness of ENLCA, we build an architecture called Efficient Non-Local Contrastive Network (ENLCN) by adding a few of our modules in a simple backbone. Extensive experimental results show that ENLCN reaches superior performance over state-of-the-art approaches on both quantitative and qualitative evaluations.

## Train
### Prepare training data 
1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Run
```shell
# Prune from 256 to 49, pr=0.80859375, x2
CUDA_VISIBLE_DEVICES=1 python main.py --model LEDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX2_DF2K_M311.pt --same_pruned_wg_criterion reg --save main/SR/LEDSR_F256R16BIX2_DF2K_ASSL0.80859375_RGP0.0001_RUL0.5_Pretrain

The code is based on EDSR.

## Test
### Quick start

The training models are available at the [link](https://drive.google.com/drive/folders/1jYdMA0ocnb-DAr71YhOduuCySOxhoAeX?usp=sharing): 


train ENLCA:

   ```
   sh demo.sh
   ```
