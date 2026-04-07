# sh feat_extractor.sh
DATA=/202331510162/llm/hx/LAION_Beyond_2B/
OUTPUT=/202331510162/llm/hx/Rememory/output/clip_feat_2B/
# oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
for SUB_CLASSES in OOP IP
do
    for DATASET in Pokemon89_39 Animals92_42 Architecture50_23 Attire54_28 FolkArt59_27 Food53_27 Insects_Spiders106_52 Landmark59_30 Plants_Fugi113_56
    do
        for SPLIT in test
        do
            python feat_extractor.py \
            --split ${SPLIT} \
            --root ${DATA} \
            --dataset-config-file ../configs/datasets/${DATASET}.yaml \
            --config-file ../configs/trainers/CoOp_openclip/vit_b16_ep200_ctxv1_lpclip.yaml \
            --output-dir ${OUTPUT} \
            --eval-only \
            DATASET.SUB_CLASSES ${SUB_CLASSES}
        done
    done
done

# sh feat_extractor.sh
DATA=/202331510162/llm/hx/LAION_Beyond_5B/
OUTPUT=/202331510162/llm/hx/Rememory/output/clip_feat_5B/
# oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
for SUB_CLASSES in OOP IP
do
    for DATASET in Pokemon89_39 Animals92_42 Architecture50_23 Attire54_28 FolkArt59_27 Food53_27 Insects_Spiders106_52 Landmark59_30 Plants_Fugi113_56
    do
        for SPLIT in test
        do
            python feat_extractor.py \
            --split ${SPLIT} \
            --root ${DATA} \
            --dataset-config-file ../configs/datasets/${DATASET}.yaml \
            --config-file ../configs/trainers/CoOp_openclip/vit_b16_ep200_ctxv1_lpclip.yaml \
            --output-dir ${OUTPUT} \
            --eval-only \
            DATASET.SUB_CLASSES ${SUB_CLASSES}
        done
    done
done
