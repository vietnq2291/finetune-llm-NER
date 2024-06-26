from model import NERModel
from tokenizer import NERTokenizer
from data import NERDataset
from serve.inference import predict

import argparse


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id")
    parser.add_argument("--description")
    parser.add_argument("--max_length", default="512")
    parser.add_argument("--output_style", default="parsed")

    args = parser.parse_args()
    model_id = args.model_id
    desc = args.description
    max_length = int(args.max_length)
    output_style = args.output_style

    # Load model
    NER_model = NERModel("inference")
    NER_model.from_pretrained(model_id, desc)

    NER_tokenizer = NERTokenizer()
    NER_tokenizer.from_pretrained(model_id, desc)

    NER_dataset = NERDataset()

    # Inference loop
    while True:
        text = input("Text: ")
        if text == "":
            break
        entity_type = input("Entity type: ")
        if entity_type == "":
            break

        inp = NER_dataset.instruction_template["input"](
            text, NER_dataset.query_template(entity_type)
        )
        out = predict(
            inp,
            NER_model.model,
            NER_tokenizer.tokenizer,
            NER_dataset.parse_output,
            output_style,
            max_length,
        )
        print("Output:", out)
        print("----------------------------------")


if __name__ == "__main__":
    main()
