import torch
import re

def predict(sample, model, tokenizer, output_style, max_length=512, verbose=False):
    test = tokenizer(sample, add_special_tokens=True)
    input_ids = torch.tensor(test['input_ids']).unsqueeze(0).to('cuda')
    attention_mask = torch.tensor(test['attention_mask']).unsqueeze(0).to('cuda')

    pred = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)[0]
    out = tokenizer.decode(pred)

    if verbose:
        print(tokenizer.decode(test['input_ids']))
        print(out)
    
    if output_style == 'formatted':
        out = out.replace('[NEWLINE]', '\n')
        out = re.sub(r"<pad> ### Response:", "", out, 1)
        out = re.sub(r"</s>$", "", out)
        out = out.strip()
    return out

