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
SUB_CLASSES=OOP

for SEED in 1
do
    DIR=/data/chenziliang/output/${DATASET}/shots_${SHOTS}/train/${TRAINER}/${CFG}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --enhanced-base ${ENHANCE} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.TaskRes.RESIDUAL_SCALE ${SCALE} \
    DATASET.SUB_CLASSES ${SUB_CLASSES}
    # fi
done