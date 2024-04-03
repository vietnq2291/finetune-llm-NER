from transformers import (
    AutoTokenizer,
    T5Tokenizer,
)


class NERTokenizer:
    def from_pretrained(self, model_id):
        if "mt5" in model_id:
            tokenizer_class = 
        else:
            model_class = AutoModel
        self.model = model_class.from_pretrained(model_id)

