#!/bin/bash

cd ..

# custom config
DATA=/mnt/SSD_500GB/chenziliang/Rememory/DATA
TRAINER=ZeroshotCLIP_openclip
# TRAINER=CoOp

SEED=1
DATASET=$1
CFG=vit_b16_ep50_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
SUB=base


DIR=output/base2new/test_${SUB}/${DATASET}/${TRAINER}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --eval-only \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi