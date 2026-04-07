#!/bin/bash

cd ..

# custom config
DATA=/data/chenziliang/LAION_Beyond/
TRAINER=CLIP_Adapter
# TRAINER=CoOp

CFG=vit_b16_ep200_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SUB_CLASSES=OOP

# animals103 insects_spiders108 plants_fungi116
for DATASET in Food53_27 Insects_Spiders106_52 Landmark59_30 Plants_Fugi113_56
do
    for SUB_CLASSES in IP All
    do
        for SEED in 1 2 3
        do
            for SHOTS in 4
            do
                COMMON_DIR=${TRAINER}/ImageA_${CFG}/seed${SEED}
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
                DATASET.SUB_CLASSES ${SUB_CLASSES}
            done
        done
    done
done