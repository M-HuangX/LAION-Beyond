#!/bin/bash

cd ..

# custom config
DATA=/data/chenziliang/LAION_Beyond/
TRAINER=ZeroshotCLIP_openclip
# TRAINER=CoOp

SEED=3
CFG=vit_b16_ep50_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
#Animals92_42
USE_SUPERCLASS_F_ZS_TEST=True

for DATASET in Animals92_42 Architecture50_23 Attire54_28 FolkArt59_27 Food53_27 Insects_Spiders106_52 Landmark59_30 Plants_Fugi113_56
do
    for SUB_CLASSES in OOP
    do
        DIR=/data/chenziliang/output/${DATASET}/${TRAINER}/ZS_${SUB_CLASSES}_superclass/seed${SEED}
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
            DATASET.SUB_CLASSES ${SUB_CLASSES} \
            DATASET.USE_SUPERCLASS_F_ZS_TEST ${USE_SUPERCLASS_F_ZS_TEST}
        fi
    done
done
