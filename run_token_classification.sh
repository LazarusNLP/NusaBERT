################################
# IndoNLU
################################

python scripts/run_token_classification.py \
    --model-checkpoint LazarusNLP/NusaBERT-base \
    --dataset-name indonlp/indonlu \
    --dataset-config posp \
    --input-column-name tokens \
    --target-column-name pos_tags \
    --output-dir outputs/nusabert-base-posp \
    --num-train-epochs 10 \
    --optim adamw_torch_fused \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 16 \
    --per-device-eval-batch-size 64 \
    --hub-model-id LazarusNLP/NusaBERT-base-POSP

python scripts/run_token_classification.py \
    --model-checkpoint LazarusNLP/NusaBERT-base \
    --dataset-name indonlp/indonlu \
    --dataset-config bapos \
    --input-column-name tokens \
    --target-column-name pos_tags \
    --output-dir outputs/nusabert-base-bapos \
    --num-train-epochs 10 \
    --optim adamw_torch_fused \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 16 \
    --per-device-eval-batch-size 64 \
    --hub-model-id LazarusNLP/NusaBERT-base-BaPOS

python scripts/run_token_classification.py \
    --model-checkpoint LazarusNLP/NusaBERT-base \
    --dataset-name indonlp/indonlu \
    --dataset-config terma \
    --input-column-name tokens \
    --target-column-name seq_label \
    --output-dir outputs/nusabert-base-terma \
    --num-train-epochs 10 \
    --optim adamw_torch_fused \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 16 \
    --per-device-eval-batch-size 64 \
    --hub-model-id LazarusNLP/NusaBERT-base-TermA

python scripts/run_token_classification.py \
    --model-checkpoint LazarusNLP/NusaBERT-base \
    --dataset-name indonlp/indonlu \
    --dataset-config keps \
    --input-column-name tokens \
    --target-column-name seq_label \
    --output-dir outputs/nusabert-base-keps \
    --num-train-epochs 10 \
    --optim adamw_torch_fused \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 16 \
    --per-device-eval-batch-size 64 \
    --hub-model-id LazarusNLP/NusaBERT-base-KEPS

python scripts/run_token_classification.py \
    --model-checkpoint LazarusNLP/NusaBERT-base \
    --dataset-name indonlp/indonlu \
    --dataset-config nergrit \
    --input-column-name tokens \
    --target-column-name ner_tags \
    --output-dir outputs/nusabert-base-nergrit \
    --num-train-epochs 10 \
    --optim adamw_torch_fused \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 16 \
    --per-device-eval-batch-size 64 \
    --hub-model-id LazarusNLP/NusaBERT-base-NERGrit

python scripts/run_token_classification.py \
    --model-checkpoint LazarusNLP/NusaBERT-base \
    --dataset-name indonlp/indonlu \
    --dataset-config nerp \
    --input-column-name tokens \
    --target-column-name ner_tags \
    --output-dir outputs/nusabert-base-nerp \
    --num-train-epochs 10 \
    --optim adamw_torch_fused \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 16 \
    --per-device-eval-batch-size 64 \
    --hub-model-id LazarusNLP/NusaBERT-base-NERP

################################
# WikiANN
################################

for lang in ace id map-bms min ms jv su
do
    python scripts/run_token_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-base \
        --dataset-name wikiann \
        --dataset-config $lang \
        --input-column-name tokens \
        --target-column-name ner_tags \
        --output-dir outputs/nusabert-base-wikiann-$lang \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 8 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-base-WikiANN-$lang
done