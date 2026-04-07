# sh feat_extractor.sh
DATA=/data/chenziliang/LAION_Beyond/
OUTPUT=/data/chenziliang/output/clip_feat2/

# oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
for DATASET in Attire54_28 FolkArt59_27 Food53_27
do
    for SPLIT in train val test
    do
        python feat_extractor.py \
        --split ${SPLIT} \
        --root ${DATA} \
        --dataset-config-file ../configs/datasets/${DATASET}.yaml \
        --config-file ../configs/trainers/CoOp_openclip/vit_b16_ep200_ctxv1_lpclip.yaml \
        --output-dir ${OUTPUT} \
        --eval-only
    done
done
