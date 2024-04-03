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
    parser.add_argument("--output_style", default="formatted")

    args = parser.parse_args()
    model_id = args.model_id
    max_length = int(args.max_length)
    output_style = args.output_style

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

        inp = NER_dataset.instruction_template['input'](
            text,
            NER_dataset.query_template(entity_type)
        )
        out = predict(inp, NER_model.model, NER_tokenizer.tokenizer, NER_dataset.get_clean_output, output_style)
        print("Output:", out)
        print("----------------------------------")

if __name__=='__main__':
    main()
