from transformers import (
    AutoModel,
    T5ForConditionalGeneration,
)


class NERModel:

    def __init__(self, usage='train'):
        if usage == 'inference':
            self.setup = self.eval
        else:
            self.setup = lambda x: None

    def from_pretrained(self, model_id):
        if "t5" in model_id:
            model_class = T5ForConditionalGeneration
        else:
            model_class = AutoModel
        self.model = model_class.from_pretrained(model_id)
        self.model.cuda()
        self.setup()

    def eval(self):
        self.model.eval()
