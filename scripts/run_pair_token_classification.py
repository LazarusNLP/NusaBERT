# Modified from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_classification.py

from dataclasses import dataclass
from datargs import parse

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


@dataclass
class Args:
    model_checkpoint: str = "LazarusNLP/NusaBERT-base"
    dataset_name: str = "indonlp/indonlu"
    dataset_config: str = "facqa"
    input_column_name_1: str = "question"
    input_column_name_2: str = "passage"
    target_column_name: str = "seq_label"
    output_dir: str = "outputs/nusabert-base-facqa"
    num_train_epochs: int = 10
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0
    optim: str = "adamw_torch_fused"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 64
    hub_model_id: str = "LazarusNLP/NusaBERT-base-FacQA"


def main(args: Args):
    # load dataset, tokenizer, model
    dataset = load_dataset(args.dataset_name, args.dataset_config, trust_remote_code=True)

    label_list = dataset["train"].features[args.target_column_name].feature.names
    num_labels = len(label_list)
    label2id = {v: i for i, v in enumerate(label_list)}
    id2label = {i: v for i, v in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_checkpoint, num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[args.input_column_name_1],
            examples[args.input_column_name_2],
            truncation=True,
            is_split_into_words=True,
        )
        tokenized_questions = tokenizer(examples[args.input_column_name_1], truncation=True, is_split_into_words=True)
        tokenized_passages = tokenizer(
            examples[args.input_column_name_2], truncation=True, is_split_into_words=True, add_special_tokens=False
        )

        labels = []
        for i, label in enumerate(examples[args.target_column_name]):
            word_ids = tokenized_passages.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            # Label question tokens as -100
            label_ids = [-100] * len(tokenized_questions["input_ids"][i])
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            # EOS [SEP] tokens for BERT
            if len(label_ids) == len(tokenized_inputs["input_ids"][i]) - 1:
                label_ids.append(-100)
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    # prepare data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # metrics for evaluation
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    callbacks = [EarlyStoppingCallback(args.early_stopping_patience, args.early_stopping_threshold)]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        optim=args.optim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=True,
        report_to="tensorboard",
        push_to_hub=True,
        hub_model_id=args.hub_model_id,
        hub_private_repo=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    trainer.evaluate(tokenized_dataset["test"])

    trainer.push_to_hub()


if __name__ == "__main__":
    args = parse(Args)
    main(args)
