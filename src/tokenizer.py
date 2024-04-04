from transformers import (
    AutoTokenizer,
    T5Tokenizer,
)


class NERTokenizer:
    def from_pretrained(self, model_id, desc):
        if "t5" in desc:
            tokenizer_class = T5Tokenizer
        else:
            tokenizer_class = AutoTokenizer
        self.tokenizer = tokenizer_class.from_pretrained(model_id)

        self.tokenizer.bos_token = self.tokenizer.pad_token
        self.tokenizer.sep_token = self.tokenizer.eos_token
