from dataclasses import dataclass

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


@dataclass
class Args:
    vocab_size: int = 10_000
    limit_alphabet: int = 100
    min_frequency: int = 4
    base_model_name: str = "indobenchmark/indobert-base-p1"
    hf_repo_id: str = "LazarusNLP/nusabert-base"


def main(args: Args):
    all_datasets = []

    # IndoWiki
    wiki_langs = ["ace", "ban", "bjn", "bug", "gor", "id", "jv", "map_bms", "min", "ms", "nia", "su", "tet"]
    wiki = load_dataset(
        "sabilmakbar/indo_wiki", "indowiki_dedup_all", split="+".join(wiki_langs), trust_remote_code=True
    )
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
    all_datasets.append(wiki)

    # KoPI-NLLB
    kopi_langs = [
        "ace_Latn-neardup",
        "ban_Latn-neardup",
        "bjn_Latn-neardup",
        "jav_Latn-neardup",
        "min_Latn-neardup",
        "sun_Latn-neardup",
    ]

    for lang in kopi_langs:
        ds = load_dataset("acul3/KoPI-NLLB", lang, split="train", trust_remote_code=True)
        ds = ds.remove_columns([col for col in ds.column_names if col != "text"])
        all_datasets.append(ds)

    # CulturaX
    culturax_langs = ["ms", "jv", "su"]
    for lang in culturax_langs:
        ds = load_dataset("uonlp/CulturaX", lang, split="train", trust_remote_code=True)
        ds = ds.remove_columns([col for col in ds.column_names if col != "text"])
        all_datasets.append(ds)

    concatenated_dataset = concatenate_datasets(all_datasets)

    def batch_iterator(batch_size=1000):
        for i in range(0, len(concatenated_dataset), batch_size):
            yield concatenated_dataset[i : i + batch_size]["text"]

    old_tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=True)

    new_tokenizer = old_tokenizer.train_new_from_iterator(
        text_iterator=batch_iterator(),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        limit_alphabet=args.limit_alphabet,
        show_progress=True,
    )

    new_tokens = set(new_tokenizer.vocab.keys()) - set(old_tokenizer.vocab.keys())
    print(len(new_tokens))

    old_tokenizer.add_tokens(list(new_tokens))
    print(len(old_tokenizer))

    old_tokenizer.push_to_hub(args.hf_repo_id, private=True)


if __name__ == "__main__":
    args = Args()
    main(args)
