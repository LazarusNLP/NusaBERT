################################
# IndoNLU
################################

python scripts/run_classification.py \
    --model-checkpoint LazarusNLP/NusaBERT-base \
    --dataset-name indonlp/indonlu \
    --dataset-config emot \
    --input-column-names tweet \
    --target-column-name label \
    --input-max-length 128 \
    --output-dir outputs/nusabert-base-emot \
    --num-train-epochs 100 \
    --optim adamw_torch_fused \
    --learning-rate 1e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 32 \
    --per-device-eval-batch-size 64 \
    --hub-model-id LazarusNLP/NusaBERT-base-EmoT

python scripts/run_classification.py \
    --model-checkpoint LazarusNLP/NusaBERT-base \
    --dataset-name indonlp/indonlu \
    --dataset-config smsa \
    --input-column-names text \
    --target-column-name label \
    --input-max-length 128 \
    --output-dir outputs/nusabert-base-smsa \
    --num-train-epochs 100 \
    --optim adamw_torch_fused \
    --learning-rate 1e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 32 \
    --per-device-eval-batch-size 64 \
    --hub-model-id LazarusNLP/NusaBERT-base-SmSA

python scripts/run_classification.py \
    --model-checkpoint LazarusNLP/NusaBERT-base \
    --dataset-name indonlp/indonlu \
    --dataset-config wrete \
    --input-column-names premise,hypothesis \
    --target-column-name label \
    --input-max-length 128 \
    --output-dir outputs/nusabert-base-wrete \
    --num-train-epochs 100 \
    --optim adamw_torch_fused \
    --learning-rate 1e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 32 \
    --per-device-eval-batch-size 64 \
    --hub-model-id LazarusNLP/NusaBERT-base-WReTE

python scripts/run_classification.py \
    --model-checkpoint LazarusNLP/NusaBERT-large \
    --dataset-name indonlp/indonlu \
    --dataset-config emot \
    --input-column-names tweet \
    --target-column-name label \
    --input-max-length 128 \
    --output-dir outputs/nusabert-large-emot \
    --num-train-epochs 100 \
    --optim adamw_torch_fused \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 16 \
    --per-device-eval-batch-size 64 \
    --hub-model-id LazarusNLP/NusaBERT-large-EmoT

python scripts/run_classification.py \
    --model-checkpoint LazarusNLP/NusaBERT-large \
    --dataset-name indonlp/indonlu \
    --dataset-config smsa \
    --input-column-names text \
    --target-column-name label \
    --input-max-length 128 \
    --output-dir outputs/nusabert-large-smsa \
    --num-train-epochs 100 \
    --optim adamw_torch_fused \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 16 \
    --per-device-eval-batch-size 64 \
    --hub-model-id LazarusNLP/NusaBERT-large-SmSA

python scripts/run_classification.py \
    --model-checkpoint LazarusNLP/NusaBERT-large \
    --dataset-name indonlp/indonlu \
    --dataset-config wrete \
    --input-column-names premise,hypothesis \
    --target-column-name label \
    --input-max-length 128 \
    --output-dir outputs/nusabert-large-wrete \
    --num-train-epochs 100 \
    --optim adamw_torch_fused \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 16 \
    --per-device-eval-batch-size 64 \
    --hub-model-id LazarusNLP/NusaBERT-large-WReTE

################################
# NusaX
################################

for lang in ace ban bbc bjn bug eng ind jav mad min nij sun
do
    python scripts/run_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-base \
        --dataset-name indonlp/NusaX-senti \
        --dataset-config $lang \
        --input-column-names text \
        --target-column-name label \
        --input-max-length 128 \
        --output-dir outputs/nusabert-base-nusax-$lang \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 1e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 32 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-base-NusaX-$lang
done

for lang in ace ban bbc bjn bug eng ind jav mad min nij sun
do
    python scripts/run_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-large \
        --dataset-name indonlp/NusaX-senti \
        --dataset-config $lang \
        --input-column-names text \
        --target-column-name label \
        --input-max-length 128 \
        --output-dir outputs/nusabert-large-nusax-$lang \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-large-NusaX-$lang
done

################################
# NusaTranslation
################################

for lang in abs btk bew bhp jav mad mak min mui rej sun
do
    python scripts/run_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-base \
        --dataset-name indonlp/nusatranslation_emot \
        --dataset-config nusatranslation_emot_$lang\_nusantara_text \
        --input-column-names text \
        --target-column-name label \
        --input-max-length 128 \
        --output-dir outputs/nusabert-base-nusatranslation-$lang \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 1e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 32 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-base-NusaTranslation-EmoT-$lang
done

for lang in abs btk bew bhp jav mad mak min mui rej sun
do
    python scripts/run_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-base \
        --dataset-name indonlp/nusatranslation_senti \
        --dataset-config nusatranslation_senti_$lang\_nusantara_text \
        --input-column-names text \
        --target-column-name label \
        --input-max-length 128 \
        --output-dir outputs/nusabert-base-nusatranslation-$lang \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 1e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 32 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-base-NusaTranslation-Senti-$lang
done

for lang in abs btk bew bhp jav mad mak min mui rej sun
do
    python scripts/run_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-large \
        --dataset-name indonlp/nusatranslation_emot \
        --dataset-config nusatranslation_emot_$lang\_nusantara_text \
        --input-column-names text \
        --target-column-name label \
        --input-max-length 128 \
        --output-dir outputs/nusabert-large-nusatranslation-$lang \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-large-NusaTranslation-EmoT-$lang
done

for lang in abs btk bew bhp jav mad mak min mui rej sun
do
    python scripts/run_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-large \
        --dataset-name indonlp/nusatranslation_senti \
        --dataset-config nusatranslation_senti_$lang\_nusantara_text \
        --input-column-names text \
        --target-column-name label \
        --input-max-length 128 \
        --output-dir outputs/nusabert-large-nusatranslation-$lang \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-large-NusaTranslation-Senti-$lang
done

################################
# NusaParagraph
################################

for lang in btk bew bug jav mad mak min mui rej sun
do
    python scripts/run_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-base \
        --dataset-name indonlp/nusaparagraph_topic \
        --dataset-config nusaparagraph_topic_$lang\_nusantara_text \
        --input-column-names text \
        --target-column-name label \
        --input-max-length 512 \
        --output-dir outputs/nusabert-base-nusaparagraph-$lang \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 1e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-base-NusaParagraph-Topic-$lang
done

for lang in btk bew bug jav mad mak min mui rej sun
do
    python scripts/run_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-base \
        --dataset-name indonlp/nusaparagraph_rhetoric \
        --dataset-config nusaparagraph_rhetoric_$lang\_nusantara_text \
        --input-column-names text \
        --target-column-name label \
        --input-max-length 512 \
        --output-dir outputs/nusabert-base-nusaparagraph-$lang \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 1e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-base-NusaParagraph-Rhetoric-$lang
done

for lang in btk bew bug jav mad mak min mui rej sun
do
    python scripts/run_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-base \
        --dataset-name indonlp/nusaparagraph_emot \
        --dataset-config nusaparagraph_emot_$lang\_nusantara_text \
        --input-column-names text \
        --target-column-name label \
        --input-max-length 512 \
        --output-dir outputs/nusabert-base-nusaparagraph-$lang \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 1e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-base-NusaParagraph-EmoT-$lang
done

for lang in btk bew bug jav mad mak min mui rej sun
do
    python scripts/run_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-large \
        --dataset-name indonlp/nusaparagraph_topic \
        --dataset-config nusaparagraph_topic_$lang\_nusantara_text \
        --input-column-names text \
        --target-column-name label \
        --input-max-length 512 \
        --output-dir outputs/nusabert-large-nusaparagraph-$lang \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-large-NusaParagraph-Topic-$lang
done

for lang in btk bew bug jav mad mak min mui rej sun
do
    python scripts/run_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-large \
        --dataset-name indonlp/nusaparagraph_rhetoric \
        --dataset-config nusaparagraph_rhetoric_$lang\_nusantara_text \
        --input-column-names text \
        --target-column-name label \
        --input-max-length 512 \
        --output-dir outputs/nusabert-large-nusaparagraph-$lang \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-large-NusaParagraph-Rhetoric-$lang
done

for lang in btk bew bug jav mad mak min mui rej sun
do
    python scripts/run_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-large \
        --dataset-name indonlp/nusaparagraph_emot \
        --dataset-config nusaparagraph_emot_$lang\_nusantara_text \
        --input-column-names text \
        --target-column-name label \
        --input-max-length 512 \
        --output-dir outputs/nusabert-large-nusaparagraph-$lang \
        --num-train-epochs 100 \
        --optim adamw_torch_fused \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-large-NusaParagraph-EmoT-$lang
done

################################
# NusaDialogue
################################

for lang in ban bug min
do
    python scripts/run_classification.py \
        --model-checkpoint LazarusNLP/NusaBERT-large \
        --dataset-name prosa-text/nusa-dialogue \
        --dataset-config $lang \
        --input-column-names dialogue \
        --target-column-name topic \
        --input-max-length 512 \
        --output-dir outputs/nusabert-large-nusadialogue-topic-$lang \
        --num-train-epochs 20 \
        --optim adamw_torch_fused \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 64 \
        --hub-model-id LazarusNLP/NusaBERT-large-NusaDialogue-Topic-$lang
done