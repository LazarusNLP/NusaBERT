# NusaBERT: Teaching IndoBERT to be multilingual and multicultural!

<div align="center">

<a href="https://huggingface.co/collections/LazarusNLP/nusabert-65dc7abe183c499cc3588b58"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collections-yellow"></img></a>

</div>

This project aims to extend the multilingual and multicultural capability of [IndoBERT](https://arxiv.org/abs/2009.05387) (Wilie et al., 2020). We expanded the IndoBERT tokenizer on 12 new regional languages of Indonesia, and continued pre-training on a large-scale corpus consisting of the Indonesian language and 12 regional languages of Indonesia. Our models are highly competitive and robust on multilingual and multicultural benchmarks, such as [IndoNLU](https://github.com/IndoNLP/indonlu), [NusaX](https://github.com/IndoNLP/nusax), and [NusaWrites](https://github.com/IndoNLP/nusa-writes).

<p align="center">
    <img src="https://raw.githubusercontent.com/LazarusNLP/NusaBERT/main/assets/logo.png" alt="logo" width="400"/>
</p>

## Pre-trained Models

| Model                                                                         | #params | Dataset                                                                                                                                                                                                              |
| ----------------------------------------------------------------------------- | :-----: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [LazarusNLP/NusaBERT-base](https://huggingface.co/LazarusNLP/NusaBERT-base)   |  111M   | [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki), [acul3/KoPI-NLLB](https://huggingface.co/datasets/acul3/KoPI-NLLB), [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) |
| [LazarusNLP/NusaBERT-large](https://huggingface.co/LazarusNLP/NusaBERT-large) |  337M   | [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki), [acul3/KoPI-NLLB](https://huggingface.co/datasets/acul3/KoPI-NLLB), [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) |

## Results

We evaluate our models on three benchmarks: [IndoNLU](https://github.com/IndoNLP/indonlu), [NusaX](https://github.com/IndoNLP/nusax), and [NusaWrites](https://github.com/IndoNLP/nusa-writes), which measures the model's natural language understanding, multilingual, and multicultural capabilities. The datasets supports a variety of languages of Indonesia.

### IndoNLU (Classification)

| Model                                                                         |   EmoT    |   SmSA    |   CASA    |   HoASA   |   WReTE   |    AVG    |
| ----------------------------------------------------------------------------- | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| mBERT                                                                         |   67.30   |   84.14   |   72.23   |   84.63   |   84.40   |   78.54   |
| XLM-MLM                                                                       |   65.75   |   86.33   |   82.17   |   88.89   |   64.35   |   77.50   |
| XLM-R Base                                                                    |   71.15   |   91.39   |   91.71   |   91.57   |   79.95   |   85.15   |
| XLM-R Large                                                                   |   78.51   |   92.35   |   92.40   | **94.27** |   83.82   |   88.27   |
| IndoBERT Lite Base p1                                                         |   73.88   |   90.85   |   89.68   |   88.07   |   82.17   |   84.93   |
| IndoBERT Lite Base p2                                                         |   72.27   |   90.29   |   87.63   |   87.62   |   83.62   |   84.29   |
| IndoBERT Base p1                                                              |   75.48   |   87.73   |   93.23   |   92.07   |   78.55   |   85.41   |
| IndoBERT Base p2                                                              |   76.28   |   87.66   |   93.24   |   92.70   |   78.68   |   85.71   |
| IndoBERT Lite Large p1                                                        |   75.19   |   88.66   |   90.99   |   89.53   |   78.98   |   84.67   |
| IndoBERT Lite Large p2                                                        |   70.80   |   88.61   |   88.13   |   91.05   | **85.41** |   84.80   |
| IndoBERT Large p1                                                             |   77.08   | **92.72** | **95.69** |   93.75   |   82.91   | **88.43** |
| IndoBERT Large p2                                                             | **79.47** |   92.03   |   94.94   |   93.38   |   80.30   |   88.02   |
| *Our work*                                                                    |
| [LazarusNLP/NusaBERT-base](https://huggingface.co/LazarusNLP/NusaBERT-base)   |   76.10   |   87.46   |   91.26   |   89.80   |   76.77   |   84.28   |
| [LazarusNLP/NusaBERT-large](https://huggingface.co/LazarusNLP/NusaBERT-large) |   78.90   |   87.36   |   92.13   |   93.18   |   82.64   |   86.84   |

### IndoNLU (Sequence Labeling)

| Model                                                                         |   POSP    |   BaPOS   |   TermA   |   KEPS    |  NERGrit  |   NERP    |   FacQA   |    AVG    |
| ----------------------------------------------------------------------------- | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| mBERT                                                                         |   91.85   |   83.25   |   89.51   |   64.31   |   75.02   |   69.27   |   61.29   |   76.36   |
| XLM-MLM                                                                       |   95.87   |   88.40   |   90.55   |   65.35   |   74.75   |   75.06   |   62.15   |   78.88   |
| XLM-R Base                                                                    |   95.16   |   84.64   |   90.99   |   68.82   |   79.09   |   75.03   |   64.58   |   79.76   |
| XLM-R Large                                                                   |   92.73   |   87.03   |   91.45   |   70.88   |   78.26   |   78.52   | **74.61** |   81.92   |
| IndoBERT Lite Base p1                                                         |   91.40   |   75.10   |   89.29   |   69.02   |   66.62   |   46.58   |   54.99   |   70.43   |
| IndoBERT Lite Base p2                                                         |   90.05   |   77.59   |   89.19   |   69.13   |   66.71   |   50.52   |   49.18   |   70.34   |
| IndoBERT Base p1                                                              |   95.26   |   87.09   |   90.73   |   70.36   |   69.87   |   75.52   |   53.45   |   77.47   |
| IndoBERT Base p2                                                              |   95.23   |   85.72   |   91.13   |   69.17   |   67.42   |   75.68   |   57.06   |   77.34   |
| IndoBERT Lite Large p1                                                        |   91.56   |   83.74   |   90.23   |   67.89   |   71.19   |   74.37   |   65.50   |   77.78   |
| IndoBERT Lite Large p2                                                        |   94.53   |   84.91   |   90.72   |   68.55   |   73.07   |   74.89   |   62.87   |   78.51   |
| IndoBERT Large p1                                                             |   95.71   |   90.35   |   91.87   |   71.18   |   77.60   |   79.25   |   62.48   |   81.21   |
| IndoBERT Large p2                                                             |   95.34   |   87.36   | **92.14** |   71.27   |   76.63   |   77.99   |   68.09   |   81.26   |
| *Our work*                                                                    |
| [LazarusNLP/NusaBERT-base](https://huggingface.co/LazarusNLP/NusaBERT-base)   |   95.77   |   96.02   |   90.54   |   66.67   |   72.93   |   82.29   |   54.81   |   79.86   |
| [LazarusNLP/NusaBERT-large](https://huggingface.co/LazarusNLP/NusaBERT-large) | **96.89** | **96.76** |   91.73   | **71.53** | **79.86** | **85.12** |   66.77   | **84.09** |

### NusaX

| Model                                                                         |  `ace`   |   `ban`   |  `bbc`   |   `bjn`   |  `bug`   |  `eng`   |   `ind`   |   `jav`   |   `mad`   |   `min`   |   `nij`   |  `sun`   |    AVG    |
| ----------------------------------------------------------------------------- | :------: | :-------: | :------: | :-------: | :------: | :------: | :-------: | :-------: | :-------: | :-------: | :-------: | :------: | :-------: |
| Naive Bayes                                                                   |   72.5   |   72.6    |   73.0   |   71.9    |   73.7   |   76.5   |   73.1    |   69.4    |   66.8    |   73.2    |   68.8    |   71.9   |   72.0    |
| SVM                                                                           |   75.7   |   75.3    | **76.7** |   74.8    |   77.2   |   75.0   |   78.7    |   71.3    |   73.8    |   76.7    |   75.1    |   74.3   |   75.4    |
| Logistic Regression                                                           |   77.4   |   76.3    |   76.3   |   75.0    | **77.2** |   75.9   |   74.7    |   73.7    |   74.7    |   74.8    |   73.4    |   75.8   |   75.4    |
| IndoNLU IndoBERT Base                                                         |   75.4   |   74.8    |   70.0   |   83.1    |   73.9   |   79.5   |   90.0    |   81.7    |   77.8    |   82.5    |   75.8    |   77.5   |   78.5    |
| IndoNLU IndoBERT Large                                                        |   76.3   |   79.5    |   74.0   |   83.2    |   70.9   |   87.3   |   90.2    |   85.6    |   77.2    |   82.9    |   75.8    |   77.2   |   80.0    |
| IndoLEM IndoBERT Base                                                         |   72.6   |   65.4    |   61.7   |   71.2    |   66.9   |   71.2   |   87.6    |   74.5    |   71.8    |   68.9    |   69.3    |   71.7   |   71.1    |
| mBERT Base                                                                    |   72.2   |   70.6    |   69.3   |   70.4    |   68.0   |   84.1   |   78.0    |   73.2    |   67.4    |   74.9    |   70.2    |   74.5   |   72.7    |
| XLM-R Base                                                                    |   73.9   |   72.8    |   62.3   |   76.6    |   66.6   |   90.8   |   88.4    |   78.9    |   69.7    |   79.1    |   75.0    |   80.1   |   76.2    |
| XLM-R Large                                                                   |   75.9   |   77.1    |   65.5   |   86.3    |   70.0   | **92.6** |   91.6    |   84.2    |   74.9    |   83.1    |   73.3    | **86.0** |   80.0    |
| *Our work*                                                                    |
| [LazarusNLP/NusaBERT-base](https://huggingface.co/LazarusNLP/NusaBERT-base)   |  76.51   |   78.67   |  74.02   |   82.38   |  71.64   |  84.09   |   89.74   |   84.09   |   75.62   |   80.77   |   74.93   |  85.21   |   79.81   |
| [LazarusNLP/NusaBERT-large](https://huggingface.co/LazarusNLP/NusaBERT-large) | **81.8** | **82.83** |  74.71   | **86.51** |  73.36   |  84.63   | **93.33** | **87.20** | **82.50** | **83.54** | **77.72** |  82.74   | **82.57** |

### NusaWrites (NusaParagraph)

| Models                                                                        |  Emotion  | Rhetorical Mode |   Topic   |
| ----------------------------------------------------------------------------- | :-------: | :-------------: | :-------: |
| Naive Bayes                                                                   |   75.51   |      37.73      |   85.06   |
| SVM                                                                           |   76.36   |      45.44      |   85.86   |
| Logistic Regression                                                           | **78.23** |      45.21      | **87.67** |
| IndoNLU IndoBERT Base                                                         |   67.12   |      47.92      |   85.87   |
| IndoNLU IndoBERT Large                                                        |   62.65   |      31.75      |   85.41   |
| IndoLEM IndoBERT Base                                                         |   66.94   |      51.93      |   84.87   |
| mBERT                                                                         |   63.15   |      50.01      |   73.82   |
| XLM-R Base                                                                    |   59.15   |      49.17      |   71.68   |
| XLM-R Large                                                                   |   67.42   |      51.57      |   83.05   |
| *Our work*                                                                    |
| [LazarusNLP/NusaBERT-base](https://huggingface.co/LazarusNLP/NusaBERT-base)   |   67.18   |      51.34      |   84.17   |
| [LazarusNLP/NusaBERT-large](https://huggingface.co/LazarusNLP/NusaBERT-large) |   71.82   |    **53.06**    |   85.89   |

### NusaWrites (NusaTranslation)

| Models                                                                        |  Emotion  | Sentiment |
| ----------------------------------------------------------------------------- | :-------: | :-------: |
| Naive Bayes                                                                   |   52.70   |   74.89   |
| SVM                                                                           |   55.08   |   76.04   |
| Logistic Regression                                                           |   56.18   |   74.89   |
| IndoNLU IndoBERT Base                                                         |   54.50   |   75.24   |
| IndoNLU IndoBERT Large                                                        |   57.80   |   77.40   |
| IndoLEM IndoBERT Base                                                         |   52.59   |   69.08   |
| mBERT                                                                         |   44.13   |   68.72   |
| XLM-R Base                                                                    |   47.02   |   68.62   |
| XLM-R Large                                                                   |   54.84   |   79.06   |
| *Our work*                                                                    |
| [LazarusNLP/NusaBERT-base](https://huggingface.co/LazarusNLP/NusaBERT-base)   |   56.54   |   77.07   |
| [LazarusNLP/NusaBERT-large](https://huggingface.co/LazarusNLP/NusaBERT-large) | **61.40** | **79.54** |

## Installation

```sh
git clone https://github.com/LazarusNLP/NusaBERT.git
cd NusaBERT
pip install -r requirements.txt
```

## Dataset

For pre-training we leverage three existing open-source corpora that includes the Indonesian language and regional languages of Indonesia. A summary of the datasets are as follows:

| Dataset                                                                        | Language               | #documents |
| ------------------------------------------------------------------------------ | ---------------------- | :--------: |
| [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)               | Indonesian (`ind`)     | 23,251,368 |
| [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)               | Javanese (`jav`)       |   2,058    |
| [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)               | Malay (`msa`)          |  238,000   |
| [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)               | Sundanese (`sun`)      |   1,554    |
| [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki) | Acehnese (`ace`)       |   12,904   |
| [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki) | Balinese (`ban`)       |   19,837   |
| [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki) | Banjarese (`bjn`)      |   10,437   |
| [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki) | Buginese (`bug`)       |   9,793    |
| [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki) | Gorontalo (`gor`)      |   14,514   |
| [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki) | Indonesian (`ind`)     |  654,287   |
| [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki) | Javanese (`jav`)       |   72,667   |
| [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki) | Banyumasan (`map_bms`) |   11,832   |
| [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki) | Minangkabau (`min`)    |  225,858   |
| [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki) | Malay (`msa`)          |  346,186   |
| [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki) | Nias (`nia`)           |   1,650    |
| [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki) | Sundanese (`sun`)      |   61,494   |
| [sabilmakbar/indo_wiki](https://huggingface.co/datasets/sabilmakbar/indo_wiki) | Tetum (`tet`)          |   1,465    |
| [acul3/KoPI-NLLB](https://huggingface.co/datasets/acul3/KoPI-NLLB)             | Acehnese (`ace`)       |  792,594   |
| [acul3/KoPI-NLLB](https://huggingface.co/datasets/acul3/KoPI-NLLB)             | Balinese (`ban`)       |  244,545   |
| [acul3/KoPI-NLLB](https://huggingface.co/datasets/acul3/KoPI-NLLB)             | Banjarese (`bjn`)      |  296,314   |
| [acul3/KoPI-NLLB](https://huggingface.co/datasets/acul3/KoPI-NLLB)             | Javanese (`jav`)       | 1,155,142  |
| [acul3/KoPI-NLLB](https://huggingface.co/datasets/acul3/KoPI-NLLB)             | Minangkabau (`min`)    |  113,323   |
| [acul3/KoPI-NLLB](https://huggingface.co/datasets/acul3/KoPI-NLLB)             | Sundanese (`sun`)      |  894,626   |

## Extend NusaBERT Tokenizer

We first need to train a WordPiece tokenizer on our pre-pretraining corpus, whose vocab size we limit up to 10,000. We then add non-overlapping tokens from the new tokenizer to the original IndoBERT tokenizer. Since there are overlapping tokens between the two tokenizers, we only ended up adding 1,511 new tokens to the original tokenizer. Refer to the [script](https://github.com/LazarusNLP/NusaBERT/blob/main/scripts/train_nusabert_tokenizer.py) for more details.

## Pre-train NusaBERT

We modified the Hugging Face 🤗 masked language modeling [pre-training script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py) and conducted continued pre-training of IndoBERT on the dataset detailed above. Running pre-training is as simple as:

```sh
python scripts/run_mlm.py \
    --model_name_or_path indobenchmark/indobert-base-p1 \
    --tokenizer_name LazarusNLP/nusabert-base \
    --max_seq_length 128 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --do_train --do_eval \
    --max_steps 500000 \
    --warmup_steps 24000 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --optim adamw_torch_fused \
    --bf16 \
    --preprocessing_num_workers 24 \
    --dataloader_num_workers 24 \
    --save_steps 10000 --save_total_limit 3 \
    --output_dir outputs/nusabert-base \
    --overwrite_output_dir \
    --report_to tensorboard \
    --push_to_hub --hub_private_repo \
    --hub_model_id LazarusNLP/nusabert-base
```

We achieved a negative log-likelihood loss of 1.4876 and an accuracy of 68.66% on a heldout subset (5%) of the pre-training corpus.

## Fine-tune NusaBERT

We developed fine-tuning scripts for NusaBERT based on fine-tuning scripts from Hugging Face 🤗's [sample fine-tuning scripts](https://github.com/huggingface/transformers/tree/main/examples/pytorch).

In particular, we developed fine-tuning scripts for single-sentence classification, multi-class multi-label classification, token classification, and pair token classification, which you can find in [scripts](https://github.com/LazarusNLP/NusaBERT/tree/main/scripts). These scripts support IndoNLU, NusaX, and NusaWrites datasets.

### Single-Sentence Classification Task

The tasks included under this category are emotion classification, sentiment analysis, topic classification, etc. To fine-tune for single-sentence classification, run the following command and modify accordingly:

```sh
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
```

Single-Sentence Classification recipes are provided [here](https://github.com/LazarusNLP/NusaBERT/blob/main/run_classification.sh).

### Multi-label Multi-class Classification

The task included under this category is aspect-based sentiment analysis (e.g. IndoNLU CASA and HoASA). To fine-tune for multi-label multi-class classification, run the following command and modify accordingly:

```sh
python scripts/run_multi_label_classification.py \
    --model-checkpoint LazarusNLP/NusaBERT-base \
    --dataset-name indonlp/indonlu \
    --dataset-config casa \
    --input-column-name sentence \
    --target-column-names fuel,machine,others,part,price,service \
    --input-max-length 128 \
    --output-dir outputs/nusabert-base-casa \
    --num-train-epochs 100 \
    --optim adamw_torch_fused \
    --learning-rate 1e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 32 \
    --per-device-eval-batch-size 64 \
    --hub-model-id LazarusNLP/NusaBERT-base-CASA
```

Multi-label Multi-class Classification recipes are provided [here](https://github.com/LazarusNLP/NusaBERT/blob/main/run_multi_label_classification.sh).

### Token Classification

Token classification is also known as sequence labeling. The tasks included under this category are part-of-speech tagging (POS), named entity recognition (NER), and token-level span extraction (e.g. IndoNLU TermA, KEPS). To fine-tune for token classification, run the following command and modify accordingly:

```sh
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
```

Token Classification recipes are provided [here](https://github.com/LazarusNLP/NusaBERT/blob/main/run_token_classification.sh).

### Pair Token Classification

Pair token classification is much like token-classification, except involving a pair of input sentences instead of one. The tasks included under this category is token-level question-passage-answering (e.g. IndoNLU FacQA). To fine-tune for pair question-answering, run the following command and modify accordingly:

```sh
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
```

Pair Token Classification recipes are provided [here](https://github.com/LazarusNLP/NusaBERT/blob/main/run_pair_token_classification.sh).

## Credits

NusaBERT is developed with love by:

<div style="display: flex;">
<a href="https://github.com/anantoj">
    <img src="https://github.com/anantoj.png" alt="GitHub Profile" style="border-radius: 50%;width: 64px;border: solid 0px #fff;margin:0 4px;">
</a>

<a href="https://github.com/DavidSamuell">
    <img src="https://github.com/DavidSamuell.png" alt="GitHub Profile" style="border-radius: 50%;width: 64px;border: solid 0px #fff;margin:0 4px;">
</a>

<a href="https://github.com/stevenlimcorn">
    <img src="https://github.com/stevenlimcorn.png" alt="GitHub Profile" style="border-radius: 50%;width: 64px;border: solid 0px #fff;margin:0 4px;">
</a>

<a href="https://github.com/w11wo">
    <img src="https://github.com/w11wo.png" alt="GitHub Profile" style="border-radius: 50%;width: 64px;border: solid 0px #fff;margin:0 4px;">
</a>
</div>