################################
# IndoNLU
################################

python scripts/run_pair_token_classification.py \
    --model-checkpoint LazarusNLP/NusaBERT-base \
    --dataset-name indonlp/indonlu \
    --dataset-config facqa \
    --input-column-name-1 question \
    --input-column-name-2 passage \
    --target-column-name seq_label \
    --output-dir outputs/nusabert-base-facqa \
    --num-train-epochs 10 \
    --optim adamw_torch_fused \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 16 \
    --per-device-eval-batch-size 64 \
    --hub-model-id LazarusNLP/NusaBERT-base-FacQA