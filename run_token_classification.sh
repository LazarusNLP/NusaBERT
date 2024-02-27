################################
# IndoNLU
################################

for size in base large
do
    python scripts/run_token_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-$size \
        --dataset-name indonlp/indonlu \
        --dataset-config posp \
        --input-column-name tokens \
        --target-column-name pos_tags \
        --output-dir outputs/nusabert-$size-posp \
        --num-train-epochs 10 \
        --optim adamw_torch_fused \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-$size-POSP

    python scripts/run_token_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-$size \
        --dataset-name indonlp/indonlu \
        --dataset-config bapos \
        --input-column-name tokens \
        --target-column-name pos_tags \
        --output-dir outputs/nusabert-$size-bapos \
        --num-train-epochs 10 \
        --optim adamw_torch_fused \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-$size-BaPOS

    python scripts/run_token_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-$size \
        --dataset-name indonlp/indonlu \
        --dataset-config terma \
        --input-column-name tokens \
        --target-column-name seq_label \
        --output-dir outputs/nusabert-$size-terma \
        --num-train-epochs 10 \
        --optim adamw_torch_fused \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-$size-TermA

    python scripts/run_token_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-$size \
        --dataset-name indonlp/indonlu \
        --dataset-config keps \
        --input-column-name tokens \
        --target-column-name seq_label \
        --output-dir outputs/nusabert-$size-keps \
        --num-train-epochs 10 \
        --optim adamw_torch_fused \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-$size-KEPS

    python scripts/run_token_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-$size \
        --dataset-name indonlp/indonlu \
        --dataset-config nergrit \
        --input-column-name tokens \
        --target-column-name ner_tags \
        --output-dir outputs/nusabert-$size-nergrit \
        --num-train-epochs 10 \
        --optim adamw_torch_fused \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-$size-NERGrit

    python scripts/run_token_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-$size \
        --dataset-name indonlp/indonlu \
        --dataset-config nerp \
        --input-column-name tokens \
        --target-column-name ner_tags \
        --output-dir outputs/nusabert-$size-nerp \
        --num-train-epochs 10 \
        --optim adamw_torch_fused \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-$size-NERP
done

################################
# WikiANN
################################

for size in base large
do
    for lang in ace id map-bms min ms jv su
    do
        python scripts/run_token_classification.py \
            --model-checkpoint LazarusNLP/NusaBERT-$size \
            --dataset-name wikiann \
            --dataset-config $lang \
            --input-column-name tokens \
            --target-column-name ner_tags \
            --output-dir outputs/nusabert-$size-wikiann-$lang \
            --num-train-epochs 100 \
            --optim adamw_torch_fused \
            --learning-rate 2e-5 \
            --weight-decay 0.01 \
            --per-device-train-batch-size 8 \
            --per-device-eval-batch-size 64 \
            --hub-model-id LazarusNLP/NusaBERT-$size-WikiANN-$lang
    done
done