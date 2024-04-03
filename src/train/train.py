from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import load_dataset
import argparse
from ..model import NERModel


def main():
    model_id = "google/mt5-base"
    ner_model = NERModel(model_id)


if __name__ == "__main__":
    main()
