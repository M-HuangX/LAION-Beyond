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

CFG=vit_b16_ep100_ctxv1_align
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=4

MAX_EPOCH=200
SAVEFREQ=10
NOTEST=False
LR=0.0001390625

BATCH_SIZE=89

DIR=/mnt/HDD_2TB/ziliang_files/output/learning2name/train/${DATASET}/FLE_${FLE}_NCTX_${NCTX}_CIFC_${CIFC}/shots_${SHOTS}/${TRAINER}/Caption_EXP2/MEPOCH_${MAX_EPOCH}_BS_${BATCH_SIZE}_LR_${LR}/seed${SEED}

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
DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH_SIZE}

# Stage2-----------------------------------------------------------------------------------
# custom config
# DATA=/home/chenziliang/Rememory/Bing_Datasets
# TRAINER=LTNa_openclip

# DATASET=$1
# SEED=$2
# # Fix length embedding
# FLE=False
# # Length of fixed length embedding
# NCTX=4
# # If FLE == False, use dynamic length embedding, whether use embeding initilized from clip
# CIFC=True

# CFG=vit_b16_ep100_ctxv1
# # CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# # CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
# SHOTS=4

# MAX_EPOCH=50
# SAVEFREQ=10
# NOTEST=False
# PRE_LR=0.005
# LR=0.0002

# STAGE2=True


# LOAD_DIR=/mnt/HDD_2TB/ziliang_files/output/learning2name/train/${DATASET}/FLE_${FLE}_NCTX_${NCTX}_CIFC_${CIFC}/shots_${SHOTS}/${TRAINER}/LR_${PRE_LR}/${CFG}/seed${SEED}

# DIR=/mnt/HDD_2TB/ziliang_files/output/learning2name/train/${DATASET}/FLE_${FLE}_NCTX_${NCTX}_CIFC_${CIFC}/shots_${SHOTS}/${TRAINER}/LR_${PRE_LR}/${CFG}/seed${SEED}/Train_again_LR_${LR}

# python train.py \
# --root ${DATA} \
# --seed ${SEED} \
# --trainer ${TRAINER} \
# --dataset-config-file configs/datasets/${DATASET}.yaml \
# --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
# --output-dir ${DIR} \
# TRAINER.LTN.FLE ${FLE} \
# TRAINER.LTN.N_CTX ${NCTX} \
# TRAINER.LTN.CIFC ${CIFC} \
# DATASET.NUM_SHOTS ${SHOTS} \
# TRAIN.CHECKPOINT_FREQ ${SAVEFREQ} \
# TEST.NO_TEST ${NOTEST} \
# OPTIM.LR ${LR} \
# OPTIM.MAX_EPOCH ${MAX_EPOCH} \
# TRAINER.LTNa.LOAD_DIR ${LOAD_DIR} \
# TRAINER.LTNa.STAGE2 ${STAGE2}
