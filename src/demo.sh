#!/bin/bash
#Train x2
#python main.py --dir_data /media/data2/xiabin/datasets --n_GPUs 1 --rgb_range 1  --lr 1e-5 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 32 --model NLSN --scale 4 --patch_size 96 --save NLSN_x4 --data_train DIV2K --data_test Set14 --test_every 500 --resume -1
CUDA_VISIBLE_DEVICES=6 python main.py --dir_data /media/data2/xiabin/datasets --n_GPUs 1 --rgb_range 1  --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 16 --model ENLCN --scale 4 --patch_size 96 --save ENLCN_x4 --data_train DIV2K --data_test Set14


#Test x2
#python main.py --dir_data ../../ --model NLSN  --chunk_size 144 --data_test Set5+Set14+B100+Urban100 --n_hashes 4 --chop --save_results --rgb_range 1 --data_range 801-900 --scale 2 --n_feats 256 --n_resblocks 32 --res_scale 0.1  --pre_train model_x2.pt --test_only
