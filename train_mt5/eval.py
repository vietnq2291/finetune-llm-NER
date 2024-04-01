from datasets import load_dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch
from collections import defaultdict


def get_pretrained_tokenizer(model_id):
    tokenizer = MT5Tokenizer.from_pretrained(model_id)
    tokenizer.bos_token = tokenizer.pad_token
    tokenizer.sep_token = tokenizer.eos_token
    return tokenizer


MAX_LENGTH = 512
model_id = "nqv2291/sft-mT5_base-NER"
output_repo = "en-conll2003-evaluation-instruction_format"
ner_tags = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}
ner_full_tags = {
    "PER": "person",
    "ORG": "organization",
    "LOC": "location",
    "MISC": "miscellaneous",
}
ner_tags_inv = {v: k for k, v in ner_tags.items()}

tokenizer = get_pretrained_tokenizer(model_id)
model = MT5ForConditionalGeneration.from_pretrained(model_id)
dataset = load_dataset("conll2003")


def convert_ner_tags_to_label(sample, ner_tags_inv):
    global no
    labels = {v: [] for v in ner_full_tags.values()}
    cur_tag = ""
    cur_entity = ""
    for token, tag_id in zip(sample["tokens"], sample["ner_tags"]):
        if tag_id % 2 == 1:
            if cur_tag != "":
                labels[cur_tag].append(cur_entity)
                cur_tag = ""
                cur_entity = ""
            cur_tag = ner_full_tags[ner_tags_inv[int(tag_id)][2:]]
            cur_entity = token
        elif tag_id != 0:
            cur_entity += " " + token
    if cur_tag != "":
        labels[cur_tag].append(cur_entity)

    return {"text": " ".join(sample["tokens"]), "label": labels}


def predict(sample, entity_type, model, tokenizer):
    sample = f"[S2S] A virtual assistant answers questions from a user based on the provided paragraph.\n\n### Instruction:\n What describes {entity_type} in the text?\n\n### Input:\nText: {sample}"
    sample += "\n\n<extra_id_0>"
    sample = sample.replace("\n", "[NEWLINE]")

    test = tokenizer(sample, add_special_tokens=True)
    input_ids = torch.tensor(test["input_ids"]).unsqueeze(0).to("cuda")
    attention_mask = torch.tensor(test["attention_mask"]).unsqueeze(0).to("cuda")

    pred = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_length=MAX_LENGTH
    )[0]
    out = tokenizer.decode(pred, skip_special_tokens=True)

    out = out.replace("### Response:[NEWLINE]", "")
    out = out.replace("[NEWLINE]", "\n")
    return eval(out)


def evaluate(sample):
    pred = {}
    for tag in ner_full_tags.values():
        pred[tag] = predict(sample, tag, model, tokenizer)
    return {"pred": pred}


def main():
    global model, tokenizer, dataset
    model.cuda()
    model.eval()

    print("> Convert NER tags to labels")
    dataset = dataset.map(
        lambda example: convert_ner_tags_to_label(example, ner_tags_inv),
        remove_columns=["tokens", "pos_tags", "chunk_tags", "ner_tags"],
    )

    print("> Make Prediction")
    test_ds = dataset["test"]
    test_ds = test_ds.map(lambda example: evaluate(example))

    # Push dataset_NER to hub
    print(f"> Pushing to hub: {output_repo}")
    test_ds.push_to_hub(output_repo)


if __name__ == "__main__":
    main()
