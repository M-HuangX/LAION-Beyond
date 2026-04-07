#!/bin/bash

cd ..

# custom config
DATA=/data/chenziliang/LAION_Beyond/
TRAINER=CLIP_Adapter
# TRAINER=CoOp

DATASET=$1
SEED=$2
CFG=vit_b16_ep200_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SUB_CLASSES=OOP

# animals103 insects_spiders108 plants_fungi116


for SHOTS in 1 2 4 8 16
do
    DIR=/data/chenziliang/output/${DATASET}/shots_${SHOTS}/train/${TRAINER}/ImageA_${CFG}/seed${SEED}
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUB_CLASSES ${SUB_CLASSES}
done

