#!/bin/bash

cd ..

# custom config
DATA=/home/chenziliang/Rememory/Bing_Datasets
TRAINER=CoCoOp_openclip
# TRAINER=CoOp

DATASET=$1
SEED=$2

CFG=vit_b16_c4_ep10_batch1_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=1

MAX_EPOCH=200
SAVEFREQ=10
NOTEST=False

DIR=/mnt/HDD_2TB/ziliang_files/output/learning2name/train/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAIN.CHECKPOINT_FREQ ${SAVEFREQ} \
    TEST.NO_TEST ${NOTEST} \
    OPTIM.MAX_EPOCH ${MAX_EPOCH}
fi