from transformers import (
    AutoModel,
    T5ForConditionalGeneration,
)
import torch


class NERModel:
    def __init__(self, usage="train"):
        if usage == "inference" or usage == "evaluation":
            self.setup = self.eval
        else:
            self.setup = lambda: None

    def from_pretrained(self, model_id):
        if "t5" in model_id:
            model_class = T5ForConditionalGeneration
        else:
            model_class = AutoModel
        model = model_class.from_pretrained(model_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.model.cuda()
        self.setup()

    def eval(self):
        self.model.eval()
