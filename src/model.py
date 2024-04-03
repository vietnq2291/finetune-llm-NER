from transformers import (
    AutoModel,
    T5ForConditionalGeneration,
)


class NERModel:
    def from_pretrained(self, model_id):
        if "mt5" in model_id:
            model_class = T5ForConditionalGeneration
        else:
            model_class = AutoModel
        self.model = model_class.from_pretrained(model_id)
