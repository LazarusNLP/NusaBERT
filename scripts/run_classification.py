# Modified from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_classification.py

from dataclasses import dataclass
from datargs import parse

import evaluate
import numpy as np
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


@dataclass
class Args:
    model_checkpoint: str = "LazarusNLP/NusaBERT-base"
    dataset_name: str = "indonlp/indonlu"
    dataset_config: str = "emot"
    input_column_names: str = "tweet"
    target_column_name: str = "label"
    text_column_delimiter: str = " "
    input_max_length: int = 128
    output_dir: str = "outputs/nusabert-base-emot"
    num_train_epochs: int = 100
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0
    optim: str = "adamw_torch_fused"
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    hub_model_id: str = "LazarusNLP/NusaBERT-base-EmoT"


def main(args: Args):
    # load dataset, tokenizer, model
    dataset = load_dataset(args.dataset_name, args.dataset_config, trust_remote_code=True)
    if args.target_column_name != "label":
        dataset = dataset.rename_column(args.target_column_name, "label")

    try:
        label_list = dataset["train"].features["label"].names
    except AttributeError:
        label_list = sorted(set(dataset["train"]["label"]))
        dataset = dataset.cast_column("label", ClassLabel(names=label_list))

    num_labels = len(label_list)
    label2id = {v: i for i, v in enumerate(label_list)}
    id2label = {i: v for i, v in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint, num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    def preprocess_function(examples):
        text_column_names = args.input_column_names.split(",")
        examples["sentence"] = examples[text_column_names[0]]
        for column in text_column_names[1:]:
            for i in range(len(examples[column])):
                examples["sentence"][i] += args.text_column_delimiter + examples[column][i]
        return tokenizer(examples["sentence"], max_length=args.input_max_length, truncation=True)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # metrics for evaluation
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        result = f1.compute(predictions=predictions, references=labels, average="macro")

        return {k: round(v, 4) for k, v in result.items()}

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
