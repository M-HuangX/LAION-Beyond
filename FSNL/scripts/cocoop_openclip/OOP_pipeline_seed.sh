#!/bin/bash

cd ..

# custom config
DATA=/data/chenziliang/LAION_Beyond/
TRAINER=CoCoOp_openclip
# TRAINER=CoOp
DATASET=$1
# SHOTS=$2
# SEED=$3

CFG=vit_b16_c4_ep10_batch1_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=4

MAX_EPOCH=10
SAVEFREQ=10
NOTEST=False
SUB_CLASSES=OOP

# # animals103 insects_spiders108 plants_fungi116


# DIR=/data/chenziliang/output/${DATASET}/shots_${SHOTS}/train/${TRAINER}_${CFG}/seed${SEED}
# python train.py \
# --root ${DATA} \
# --seed ${SEED} \
# --trainer ${TRAINER} \
# --dataset-config-file configs/datasets/${DATASET}.yaml \
# --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
# --output-dir ${DIR} \
# DATASET.NUM_SHOTS ${SHOTS} \
# TRAIN.CHECKPOINT_FREQ ${SAVEFREQ} \
# TEST.NO_TEST ${NOTEST} \
# OPTIM.MAX_EPOCH ${MAX_EPOCH} \
# DATASET.SUB_CLASSES ${SUB_CLASSES}



# custom config
DATA=/data/chenziliang/LAION_Beyond/
TRAINER=CoOp_openclip
# TRAINER=CoOp

MAX_EPOCH=200

CTP=end
NCTX=4
CSC=False
CFG=vit_b16_ep200_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SEED=3

# animals103 insects_spiders108 plants_fungi116
for SHOTS in 1 2 4 8 16
do
    DIR=/data/chenziliang/output/${DATASET}/shots_${SHOTS}/train/${TRAINER}/CSC_${CSC}/${CFG}/seed${SEED}
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUB_CLASSES ${SUB_CLASSES}
done
