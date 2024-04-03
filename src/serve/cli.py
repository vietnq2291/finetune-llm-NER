from model import NERModel
from tokenizer import NERTokenizer
from data import NERDataset
from .inference import predict

import argparse


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id")
    parser.add_argument("--max_length", default="512")


    args = parser.parse_args()
    model_id = args.model_id
    max_length = int(args.max_length)

    # Load model
    NER_model = NERModel('inference')
    NER_model.from_pretrained(model_id)

    NER_tokenizer = NERTokenizer()
    NER_tokenizer.from_pretrained(model_id)

    NER_dataset = NERDataset()

    # Inference loop
    while True:
        text = input('Text: ')
        if text == "": break
        entity_type = input("Entity type: ")
        if entity_type == "": break

        inp = NER_dataset.instruction_template.input(
            text,
            NER_dataset.query_template(entity_type)
        )
        out = predict(text, entity_type, NER_model.model, NER_tokenizer.tokenizer)
        print("Output:", out)
        print("----------------------------------")

if __name__=='__main__':
    main()
