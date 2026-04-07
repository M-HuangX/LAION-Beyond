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

# Function to get num_classname from JSON file
get_num_classname() {
  local subfield=$1
  local json_file="/202331510162/llm/hx/LAION_Beyond_2B/${subfield}_clustering_results.json"
  
  # Use jq to parse JSON and count unique classnames
  local num_classname=$(jq '[.[] | .pseudo_classname] | unique | length' "$json_file")
  
  echo "$num_classname"
}

# Helper functions

DATA=/202331510162/llm/hx/LAION_Beyond_2B/
TRAINER=FSNL_openclip_GCD

SUBFIELD=$1  # Now we use SUBFIELD instead of DATASET
SEED=1

# Fix length embedding
FLE=False
# Length of fixed length embedding
NCTX=4
# If FLE == False, use dynamic length embedding, whether use embeding initilized from clip
CIFC=True
CFG=vit_b16_ep200_standard
SHOTS=16
MAX_EPOCH=100
SAVEFREQ=0
NOTEST=False

#-----------------
USE_CAPTION=False
SUB_CLASSES=OOP
#-----------------

for SHOTS in 16
do
    # Get num_classname from JSON file
    num_classname=$(get_num_classname ${SUBFIELD})
    
    # Calculate batch size and LR
    BATCH_SIZE=$(calculate_batch_size ${num_classname} ${SHOTS})
    LR=$(calculate_lr ${BATCH_SIZE})
    
    DIR=/202331510162/llm/hx/Rememory/output/LAION_Beyond_2B_GCD/${SUBFIELD}/shots_${SHOTS}/train/${TRAINER}_USE_CAPTION_${USE_CAPTION}/MEPOCH_${MAX_EPOCH}_BS_${BATCH_SIZE}_LR_${LR}/seed${SEED}

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/LAION_Beyond_GCD.yaml \
    --config-file configs/trainers/FSNL_openclip/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.FSNL.FLE ${FLE} \
    TRAINER.FSNL.N_CTX ${NCTX} \
    TRAINER.FSNL.CIFC ${CIFC} \
    TRAINER.FSNL.USE_CAPTION ${USE_CAPTION} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAIN.CHECKPOINT_FREQ ${SAVEFREQ} \
    TEST.NO_TEST ${NOTEST} \
    TEST.EVALUATOR ClusteringAccuracy \
    OPTIM.LR ${LR} \
    OPTIM.MAX_EPOCH ${MAX_EPOCH} \
    DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH_SIZE} \
    DATASET.SUB_CLASSES ${SUB_CLASSES} \
    DATASET.SUBFIELD ${SUBFIELD}
    
done