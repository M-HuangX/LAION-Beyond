feature_dir=/data/chenziliang/output/clip_feat2/

for DATASET in Pokemon89_39 Animals92_42 Architecture50_23
do
    python linear_probe.py \
    --dataset ${DATASET} \
    --feature_dir ${feature_dir} \
    --num_step 8 \
    --num_run 10
done
