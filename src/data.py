from datasets import load_dataset
import json

class NERDataset:

    def __init__(self):
        self.system_prompt = 'A virtual assistant answers questions from a user based on the provided paragraph.'
        self.query_template = lambda entity_type: f'What describes {entity_type} in the text?'
        self.instruction_template = {
            'input': lambda text, query: (
                "[S2S] "
                + self.system_prompt
                + "\n\n### Instruction:\n"
                + query
                + "\n\n### Input:\n"
                + text
                + "\n\n <extra_id_0>"
            ).replace("\n", "[NEWLINE]"),
            'label': lambda target: ("### Response:\n" + target).replace("\n", "[NEWLINE]"),
        }

    def load_dataset(self, path, data_files=None, split=None):
        self.dataset = load_dataset(path=path, data_files=data_files, split=split)

    def conversations_to_instruction(self, example):
        text = example['conversations'][0]['value']
        query = example['conversations'][2]['value']
        target = example['conversations'][-1]['value']

        instruction_example = {
            'id': example['id'],
            'input': self.instruction_template['input'](text, query),
            'label': self.instruction_template['label'](target)
        }

        return instruction_example

    def convert_dataset(self, from_format, to_format, drop_old_columns=True):
        if from_format == 'conversations' and to_format == 'instruction':
            self.dataset = self.dataset.map(self.conversations_to_instruction, remove_columns=self.dataset.column_names)


if __name__=='__main__':
    dataset = NERDataset()
    dataset.load_dataset(path='json', data_files='eval/test_data/CrossNER_AI.json', split='train')
    dataset.convert_dataset('conversations', 'instruction')

