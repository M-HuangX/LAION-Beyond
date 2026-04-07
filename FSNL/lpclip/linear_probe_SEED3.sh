feature_dir=/data/chenziliang/output/clip_feat2/

for DATASET in Insects_Spiders106_52 Landmark59_30 Plants_Fugi113_56
do
    python linear_probe.py \
    --dataset ${DATASET} \
    --feature_dir ${feature_dir} \
    --num_step 8 \
    --num_run 10
done
