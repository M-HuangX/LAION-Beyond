#!/bin/bash

cd ..


# # custom config
# DATA=/home/chenziliang/Rememory/Bing_Datasets
# TRAINER=CoOp_openclip
# # TRAINER=CoOp


# SEED=1
# CTP=end
# NCTX=4
# CSC=True
# CFG=vit_b16_ep200_ctxv1
# # CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# # CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
# SHOTS=16

# for DATASET in animals103 insects_spiders108 plants_fungi116
# do
#     DIR=output/OOP/COOP_EXP/${DATASET}/shots_${SHOTS}/${TRAINER}/CSC_${CSC}/${CFG}/seed${SEED}
#     python train.py \
#     --root ${DATA} \
#     --seed ${SEED} \
#     --trainer ${TRAINER} \
#     --dataset-config-file configs/datasets/${DATASET}.yaml \
#     --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#     --output-dir ${DIR} \
#     TRAINER.COOP.N_CTX ${NCTX} \
#     TRAINER.COOP.CSC ${CSC} \
#     TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
#     DATASET.NUM_SHOTS ${SHOTS}
# done



# custom config
DATA=/home/chenziliang/Rememory/Bing_Datasets
TRAINER=CoCoOp_openclip
# TRAINER=CoOp

SEED=1

CFG=vit_b16_c4_ep10_batch1_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=4

MAX_EPOCH=100
SAVEFREQ=10
NOTEST=False
# animals103 insects_spiders108 plants_fungi116
for DATASET in pokemon
do
    DIR=/mnt/HDD_2TB/ziliang_files/output/learning2name/train/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
done


# custom config
DATA=/home/chenziliang/Rememory/Bing_Datasets
TRAINER=CoOp_openclip
# TRAINER=CoOp


SEED=1
CTP=end
NCTX=4
CSC=False
CFG=vit_b16_ep200_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=4
# animals103 insects_spiders108 plants_fungi116
for DATASET in pokemon
do
    DIR=output/OOP/COOP_EXP/${DATASET}/shots_${SHOTS}/${TRAINER}/CSC_${CSC}/${CFG}/seed${SEED}
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
    DATASET.NUM_SHOTS ${SHOTS}
done

