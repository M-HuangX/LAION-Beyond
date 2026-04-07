feature_dir=/data/chenziliang/output/clip_feat2/

for DATASET in Attire54_28 FolkArt59_27 Food53_27
do
    python linear_probe.py \
    --dataset ${DATASET} \
    --feature_dir ${feature_dir} \
    --num_step 8 \
    --num_run 10
done
