#!/bin/bash

cd ..

# custom config
DATA=/data/chenziliang/LAION_Beyond/
TRAINER=ZeroshotCLIP_openclip
# TRAINER=CoOp

SEED=3
CFG=vit_b16_ep50_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet


for DATASET in Landmark59_30
do
    for SUB_CLASSES in All
    do
        DIR=/data/chenziliang/output/${DATASET}/${TRAINER}/ZS_${SUB_CLASSES}/seed${SEED}
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --eval-only \
        DATASET.SUB_CLASSES ${SUB_CLASSES}
    done
done
