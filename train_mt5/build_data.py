from datasets import load_dataset
import argparse
from utils import (
    get_pretrained_tokenizer,
    format_chat_template,
    make_labelled_data,
    tokenize,
    convert_alpaca_to_conversations,
)


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="google/mt5-base")
    parser.add_argument(
        "--action", choices=["push_to_hub", "save_to_disk"], required=True
    )
    parser.add_argument("--raw_dataset_path")
    parser.add_argument("--dataset_output_dir")
    parser.add_argument("--dataset_repo")
    parser.add_argument("--split_ratio", default=0.02)
    parser.add_argument("--random_seed", default=42)

    args = parser.parse_args()
    model_id = args.model_id
    raw_dataset_path = args.raw_dataset_path
    output_action = args.action
    split_ratio = args.split_ratio
    random_seed = args.random_seed

    # init tokenizer
    tokenizer = get_pretrained_tokenizer(model_id)

    print("> Loading dataset and formatting chat style to sharegpt...")
    raw_ds = load_dataset(raw_dataset_path, split="train")
    if "alpaca" in raw_dataset_path:
        chat_style_ds = raw_ds.map(
            convert_alpaca_to_conversations,
            with_indices=True,
            remove_columns=["instruction", "input", "output"],
        )
    else:
        chat_style_ds = raw_ds.map(
            lambda x: format_chat_template(tokenizer, x), remove_columns="conversations"
        )

    print(f"> Split dataset with ratio {split_ratio}:")
    splitted_ds = chat_style_ds.train_test_split(
        test_size=split_ratio, seed=random_seed
    )
    print("\tNo. train samples:", splitted_ds.shape["train"][0])
    print("\tNo. test samples: ", splitted_ds.shape["test"][0])

    print("> Making labelled dataset...")
    labelled_ds = splitted_ds.map(
        make_labelled_data, batched=True, remove_columns=chat_style_ds.column_names
    )
    print("labelled ds:", labelled_ds.column_names)

    labelled_ds.push_to_hub(args.dataset_repo)

    print("> Tokenizing dataset...")
    dataset = labelled_ds.map(
        lambda x: tokenize(x, tokenizer, "text", "label"),
        batched=True,
        remove_columns=["text", "label"],
    )

    print(">> Dataset is ready:")
    print(dataset)

    # Save dataset
    if output_action == "push_to_hub":
        output_repo = args.dataset_repo
        print(f"> Pushing to hub: {output_repo}")
        dataset.push_to_hub(output_repo)
    elif output_action == "save_to_disk":
        output_dir = args.dataset_output_dir
        print(f"> Saving to disk: {output_dir}")
        dataset.save_to_disk(output_dir)
    print(">> Finished!")


if __name__ == "__main__":
    main()
