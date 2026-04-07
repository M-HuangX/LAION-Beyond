#!/bin/bash

cd ..

# custom config
DATA=/data/chenziliang/LAION_Beyond/
TRAINER=CoCoOp_openclip
# TRAINER=CoOp

CFG=vit_b16_c4_ep10_batch1_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet

MAX_EPOCH=10
SAVEFREQ=10
NOTEST=False
SUB_CLASSES=IP

SHOTS=4
# animals103 insects_spiders108 plants_fungi116
for SEED in 1 2 3
do
    for DATASET in Pokemon89_39 Animals92_42 Architecture50_23 Attire54_28 FolkArt59_27 Food53_27 Insects_Spiders106_52 Landmark59_30 Plants_Fugi113_56
    do
        COMMON_DIR=${TRAINER}_${CFG}/seed${SEED}
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
        DATASET.NUM_SHOTS ${SHOTS} \
        OPTIM.MAX_EPOCH ${MAX_EPOCH} \
        DATASET.SUB_CLASSES ${SUB_CLASSES}
    done
done


# # custom config
# DATA=/data/chenziliang/LAION_Beyond/
# TRAINER=CoOp_openclip
# # TRAINER=CoOp


# CTP=end
# NCTX=4
# CSC=False
# CFG=vit_b16_ep200_ctxv1
# # CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# # CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet

# # animals103 insects_spiders108 plants_fungi116
# for SEED in 1 2 3
# do
#     for SHOTS in 4
#     do
#         COMMON_DIR=/${TRAINER}/CSC_${CSC}/${CFG}/seed${SEED}
#         MODEL_DIR=/data/chenziliang/output/${DATASET}/shots_${SHOTS}/train/${COMMON_DIR}
#         DIR=/data/chenziliang/output/${DATASET}/shots_${SHOTS}/test_${SUB_CLASSES}/${COMMON_DIR}
#         python train.py \
#         --root ${DATA} \
#         --seed ${SEED} \
#         --trainer ${TRAINER} \
#         --dataset-config-file configs/datasets/${DATASET}.yaml \
#         --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#         --output-dir ${DIR} \
#         TRAINER.COOP.N_CTX ${NCTX} \
#         TRAINER.COOP.CSC ${CSC} \
#         TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
#         DATASET.NUM_SHOTS ${SHOTS} \
#         DATASET.SUB_CLASSES ${SUB_CLASSES}
#     done
# done
