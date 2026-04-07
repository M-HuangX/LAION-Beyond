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
num_classname_mapping["pokemon"]=89 # Replace with your actual num_classname mapping
num_classname_mapping["animals103"]=103  
num_classname_mapping["insects_spiders108"]=108
num_classname_mapping["plants_fungi116"]=116
num_classname_mapping["folkart56"]=56

# Helper functions

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
SHOTS=4

MAX_EPOCH=200
SAVEFREQ=10
NOTEST=False

# # Extract num_classname from DATASET
# num_classname=${DATASET//[!0-9]/}
for SHOTS in 1 2 8 16
do
    # Auto-compute BATCH_SIZE and LR based on DATASET
    # Extract num_classname from DATASET using num_classname_mapping
    num_classname=${num_classname_mapping[$DATASET]}
    # Calculate batch size and LR
    BATCH_SIZE=$(calculate_batch_size ${num_classname} ${SHOTS})
    LR=$(calculate_lr ${BATCH_SIZE})
    # Auto-compute BATCH_SIZE and LR based on DATASET

    # LR=0.00064375
    # BATCH_SIZE=412

    MIX=True

    VERSION=Base
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
done