#!/bin/bash

#source activate mn

if [ -z "$1" ] 
    then
        GPU_NUM=0
else
    GPU_NUM=$1
fi

echo 'GPU NUM: '$GPU_NUM

CUDA_VISIBLE_DEVICES=$GPU_NUM CUDA_CACHE_PATH=/st2/hayeon/tmp python train_one_shot_learning_matching_network.py \
    --batch_size 32 \
    --experiment_title omniglot_20_1_matching_network \
    --total_epochs 200 --full_context_unroll_k 5 \
    --classes_per_set 20 \
    --samples_per_class 1 \
    --use_full_context_embeddings False \
    --use_mean_per_class_embeddings False \
    --dropout_rate_value 0.0 \
    --data_path ../../data/omniglot

