from model import NERModel
from tokenizer import NERTokenizer
from data import NERDataset
from serve import inference

import argparse


class NEREvaluator:
    def evaluate(self, preds, labels):
        return "evaluation result"


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id")
    parser.add_argument("--max_length", default="512")

    args = parser.parse_args()
    model_id = args.model_id
    max_length = int(args.max_length)

    # Load model
    NER_model = NERModel("evaluation")
    NER_model.from_pretrained(model_id)

    NER_tokenizer = NERTokenizer()
    NER_tokenizer.from_pretrained(model_id)

    eval_ds = NERDataset()
    eval_ds.load_dataset(
        path="json", data_files="test_data/CrossNER_AI.json", split="train"
    )
    eval_ds.convert_dataset("conversations", "instruction")

    print(eval_ds.dataset[0])
