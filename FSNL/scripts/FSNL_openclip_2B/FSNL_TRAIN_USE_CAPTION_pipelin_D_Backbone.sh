#!/bin/bash

cd ..

# Function to calculate batch size based on num_classname
calculate_batch_size() {
  local num_classname=$1
  local shots=$2

  if (( shots >= 4 )); then
    echo $(( 4 * num_classname ))
  else
    echo $(( shots * num_classname ))
  fi
}

# Function to calculate LR based on batch size
calculate_lr() {
  local batch_size=$1

  echo "scale=8; $batch_size / 128 * 0.0002" | bc
}

# Mapping of DATASET to num_classname
declare -A num_classname_mapping
num_classname_mapping["Pokemon89_39"]=89 # Replace with your actual num_classname mapping
num_classname_mapping["Animals92_42"]=92 
num_classname_mapping["Architecture50_23"]=50
num_classname_mapping["Attire54_28"]=54
num_classname_mapping["FolkArt59_27"]=59
num_classname_mapping["Food53_27"]=53
num_classname_mapping["Insects_Spiders106_52"]=106
num_classname_mapping["Landmark59_30"]=59
num_classname_mapping["Plants_Fugi113_56"]=113

# Helper functions

DATA=/data/chenziliang/LAION_Beyond/
TRAINER=FSNL_openclip_2B

DATASET=Pokemon89_39

# Fix length embedding
FLE=False
# Length of fixed length embedding
NCTX=4
# If FLE == False, use dynamic length embedding, whether use embeding initilized from clip
CIFC=True
CFG=vit_b16_ep200_standard
SHOTS=16
MAX_EPOCH=200
SAVEFREQ=10
NOTEST=False


#-----------------
USE_CAPTION=True
SUB_CLASSES=OOP
#-----------------

# # Extract num_classname from DATASET
# num_classname=${DATASET//[!0-9]/}
for SEED in 1 2 3
do
    for SHOTS in 4
    do
        # Auto-compute BATCH_SIZE and LR based on DATASET
        # Extract num_classname from DATASET using num_classname_mapping
        num_classname=${num_classname_mapping[$DATASET]}
        # Calculate batch size and LR
        BATCH_SIZE=$(calculate_batch_size ${num_classname} ${SHOTS})
        LR=$(calculate_lr ${BATCH_SIZE})
        # Auto-compute BATCH_SIZE and LR based on DATASET
        DIR=/data/chenziliang/output/${DATASET}/shots_${SHOTS}/train/${TRAINER}_USE_CAPTION_${USE_CAPTION}/MEPOCH_${MAX_EPOCH}_BS_${BATCH_SIZE}_LR_${LR}/seed${SEED}

        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.FSNL.FLE ${FLE} \
        TRAINER.FSNL.N_CTX ${NCTX} \
        TRAINER.FSNL.CIFC ${CIFC} \
        TRAINER.FSNL.USE_CAPTION ${USE_CAPTION} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAIN.CHECKPOINT_FREQ ${SAVEFREQ} \
        TEST.NO_TEST ${NOTEST} \
        OPTIM.LR ${LR} \
        OPTIM.MAX_EPOCH ${MAX_EPOCH} \
        DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH_SIZE} \
        DATASET.SUB_CLASSES ${SUB_CLASSES}

    done
done



#-----------------
USE_CAPTION=True
SUB_CLASSES=All
#-----------------

# # Extract num_classname from DATASET
# num_classname=${DATASET//[!0-9]/}
for SEED in 1 2 3
do
    for SHOTS in 4
    do
        # Auto-compute BATCH_SIZE and LR based on DATASET
        # Extract num_classname from DATASET using num_classname_mapping
        num_classname=${num_classname_mapping[$DATASET]}
        # Calculate batch size and LR
        BATCH_SIZE=$(calculate_batch_size ${num_classname} ${SHOTS})
        LR=$(calculate_lr ${BATCH_SIZE})
        # Auto-compute BATCH_SIZE and LR based on DATASET
    
        COMMON_DIR=${TRAINER}_USE_CAPTION_${USE_CAPTION}/MEPOCH_${MAX_EPOCH}_BS_${BATCH_SIZE}_LR_${LR}/seed${SEED}
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
        TRAINER.FSNL.FLE ${FLE} \
        TRAINER.FSNL.N_CTX ${NCTX} \
        TRAINER.FSNL.CIFC ${CIFC} \
        TRAINER.FSNL.USE_CAPTION ${USE_CAPTION} \
        DATASET.NUM_SHOTS ${SHOTS} \
        OPTIM.LR ${LR} \
        OPTIM.MAX_EPOCH ${MAX_EPOCH} \
        DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH_SIZE} \
        DATASET.SUB_CLASSES ${SUB_CLASSES}

    done
done