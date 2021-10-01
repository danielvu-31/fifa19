#!/bin/bash

#CUDA_VISIBLE_DEVICES=1 \
python -m tune \
    --tune_config_path config/lightgbm_bayesian.yaml \
    --data_folder data \
    --tune_result tune_result \
    --model lightgbm \
    --ray_results ray_results
