#!/bin/bash

cd ..

# custom config
DATA=/home/chenziliang/Rememory/Bing_Datasets
TRAINER=LTN_openclip

DATASET=$1
SEED=$2
# Fix length embedding
FLE=False
# Length of fixed length embedding
NCTX=4
# If FLE == False, use dynamic length embedding, whether use embeding initilized from clip
CIFC=True

CFG=vit_b16_ep100_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=16

LR=0.0002

LOADEP=20

COMMON_DIR=${DATASET}/FLE_${FLE}_NCTX_${NCTX}_CIFC_${CIFC}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=/mnt/HDD_2TB/ziliang_files/output/learning2name/train/${COMMON_DIR}
DIR=/mnt/HDD_2TB/ziliang_files/output/learning2name/test/${COMMON_DIR}/loadep${LOADEP}

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
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    TRAINER.LTN.FLE ${FLE} \
    TRAINER.LTN.N_CTX ${NCTX} \
    TRAINER.LTN.CIFC ${CIFC} \
    DATASET.NUM_SHOTS ${SHOTS}
fi