#!/bin/bash

cd ..
# Stage1
DATA=/home/chenziliang/Rememory/Bing_Datasets
TRAINER=LTNa_openclip

DATASET=$1
SEED=$2
# Fix length embedding
FLE=False
# Length of fixed length embedding
NCTX=4
# If FLE == False, use dynamic length embedding, whether use embeding initilized from clip
CIFC=True

CFG=vit_b16_ep100_ctxv1_cmxb
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=8

MAX_EPOCH=200
SAVEFREQ=10
NOTEST=False
LR=0.00064375

BATCH_SIZE=412

MIX=True

VERSION=DBC
# Base, DB, BC, DBC


DIR=/mnt/HDD_2TB/ziliang_files/output/learning2name/train/${DATASET}/FLE_${FLE}_NCTX_${NCTX}_CIFC_${CIFC}/shots_${SHOTS}/${TRAINER}/VERSION_${VERSION}/MEPOCH_${MAX_EPOCH}_BS_${BATCH_SIZE}_LR_${LR}_MIX_${MIX}/seed${SEED}

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
TRAINER.LTN.FLE ${FLE} \
TRAINER.LTN.N_CTX ${NCTX} \
TRAINER.LTN.CIFC ${CIFC} \
DATASET.NUM_SHOTS ${SHOTS} \
TRAIN.CHECKPOINT_FREQ ${SAVEFREQ} \
TEST.NO_TEST ${NOTEST} \
OPTIM.LR ${LR} \
OPTIM.MAX_EPOCH ${MAX_EPOCH} \
DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH_SIZE} \
TRAINER.LTNa.MIX ${MIX} \
TRAINER.LTNa.VERSION ${VERSION}
