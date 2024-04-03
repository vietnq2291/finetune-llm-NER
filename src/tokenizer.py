from transformers import (
    AutoTokenizer,
    T5Tokenizer,
)


class NERTokenizer:
    def from_pretrained(self, model_id):
        if "mt5" in model_id:
            tokenizer_class = T5Tokenizer
        else:
            model_class = AutoModel
        self.tokenizer = model_class.from_pretrained(model_id)

        self.tokenizer.bos_token = tokenizer.pad_token
        self.tokenizer.sep_token = tokenizer.eos_token

