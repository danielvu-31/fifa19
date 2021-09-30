#!/bin/bash

#CUDA_VISIBLE_DEVICES=1 \
python -m train \
    --data_folder data \
    --best_config_folder tune_result \
    --ckpt_folder best_ckpt \
    --model lightgbm \
    --tuning bohb
