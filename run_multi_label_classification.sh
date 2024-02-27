################################
# IndoNLU
################################

for size in base large
do
    python scripts/run_multi_label_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-$size \
        --dataset-name indonlp/indonlu \
        --dataset-config casa \
        --input-column-name sentence \
        --target-column-names fuel,machine,others,part,price,service \
        --input-max-length 128 \
        --output-dir outputs/nusabert-$size-casa \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 1e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 32 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-$size-CASA

    python scripts/run_multi_label_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-$size \
        --dataset-name indonlp/indonlu \
        --dataset-config hoasa \
        --input-column-name sentence \
        --target-column-names ac,air_panas,bau,general,kebersihan,linen,service,sunrise_meal,tv,wifi \
        --input-max-length 128 \
        --output-dir outputs/nusabert-$size-hoasa \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 1e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 32 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-$size-HoASA
done