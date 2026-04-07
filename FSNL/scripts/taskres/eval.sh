#!/bin/bash

cd ..

# custom config
DATA=/data/chenziliang/LAION_Beyond/
# DATA=/path/to/datasets
TRAINER=TaskRes

DATASET=$1
CFG=$2      # config file
ENHANCE=$3  # path to enhanced base weights
SHOTS=$4    # number of shots (1, 2, 4, 8, 16)
SCALE=$5    # scaling factor

for SUB_CLASSES in IP All
do
    for SEED in 1
    do
        COMMON_DIR=${TRAINER}/${CFG}/seed${SEED}
        MODEL_DIR=/data/chenziliang/output/${DATASET}/shots_${SHOTS}/train/${COMMON_DIR}
        DIR=/data/chenziliang/output/${DATASET}/shots_${SHOTS}/test_${SUB_CLASSES}/${COMMON_DIR}
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --eval-only \
        --enhanced-base ${ENHANCE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.TaskRes.RESIDUAL_SCALE ${SCALE} \
        DATASET.SUB_CLASSES ${SUB_CLASSES}
done
done