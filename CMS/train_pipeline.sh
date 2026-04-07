#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -m methods.contrastive_meanshift_training_openclip  \
            --dataset_name 'laion_beyond' \
            --lr 0.05 \
            --temperature 0.25 \
            --subfield 'Animals'


CUDA_VISIBLE_DEVICES=1 python -m methods.contrastive_meanshift_training_openclip  \
            --dataset_name 'laion_beyond' \
            --lr 0.05 \
            --temperature 0.25 \
            --subfield 'Architecture'


CUDA_VISIBLE_DEVICES=2 python -m methods.contrastive_meanshift_training_openclip  \
            --dataset_name 'laion_beyond' \
            --lr 0.05 \
            --temperature 0.25 \
            --subfield 'Attire'



CUDA_VISIBLE_DEVICES=3 python -m methods.contrastive_meanshift_training_openclip  \
            --dataset_name 'laion_beyond' \
            --lr 0.05 \
            --temperature 0.25 \
            --subfield 'FolkArt'


CUDA_VISIBLE_DEVICES=4 python -m methods.contrastive_meanshift_training_openclip  \
            --dataset_name 'laion_beyond' \
            --lr 0.05 \
            --temperature 0.25 \
            --subfield 'Food'


CUDA_VISIBLE_DEVICES=5 python -m methods.contrastive_meanshift_training_openclip  \
            --dataset_name 'laion_beyond' \
            --lr 0.05 \
            --temperature 0.25 \
            --subfield 'Insects_Spiders'


CUDA_VISIBLE_DEVICES=6 python -m methods.contrastive_meanshift_training_openclip  \
            --dataset_name 'laion_beyond' \
            --lr 0.05 \
            --temperature 0.25 \
            --subfield 'Landmark'


CUDA_VISIBLE_DEVICES=7 python -m methods.contrastive_meanshift_training_openclip  \
            --dataset_name 'laion_beyond' \
            --lr 0.05 \
            --temperature 0.25 \
            --subfield 'Plants_Fugi'


CUDA_VISIBLE_DEVICES=5 python -m methods.contrastive_meanshift_training_openclip  \
            --dataset_name 'laion_beyond' \
            --lr 0.05 \
            --temperature 0.25 \
            --subfield 'Pokemon'
