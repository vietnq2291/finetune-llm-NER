from model import NERModel
from tokenizer import NERTokenizer
from data import NERDataset
from serve.inference import predict
from tqdm import tqdm

import argparse


class NEREvaluator:
    def __init__(self, model, tokenizer, eval_ds):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_ds = eval_ds

    def run(self):
        # Run predictions
        preds = []
        for sample in tqdm(eval_ds.dataset):
            out = predict(
                sample["input"],
                self.model,
                self.tokenizer,
                self.eval_ds.parse_output,
                "parsed",
            )
            preds.append(out)

        # Run evaluation

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

    # Prepare data
    eval_ds = NERDataset()
    eval_ds.load_dataset(
        path="json", data_files="eval/test_data/CrossNER_AI.json", split="train"
    )
    eval_ds.convert_dataset("conversations", "instruction")

    # Create evaluator
    evaluator = NEREvaluator(NER_model.model, NER_tokenizer.tokenizer, eval_ds)

    # Run evaluation
