import torch
import json
import pandas as pd
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoTokenizer

class CustomDataset(Dataset):
    def __init__(self, file_path, has_title=True):
        self.df = pd.read_json(file_path, lines=True)
        self.has_title = has_title
        self.bodies = self.process_bodies()
        self.titles = self.process_titles() if has_title else None

    def process_bodies(self):
        bodies = self.df['body'].fillna('')
        return bodies

    def process_titles(self):
        titles = self.df['title'].fillna('')
        return titles

    def __getitem__(self, index):
        body = self.bodies[index]
        title = self.titles[index] if self.titles is not None else ''
        if self.has_title == False:
            return body
        return body, title

    def __len__(self):
        return len(self.df)
    
# load model bin for t5-small
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
model.load_state_dict(torch.load('./model/checkpoint-8200/pytorch_model.bin'))

# Usage example
test_dataset = CustomDataset('./hw5_dataset/test.jsonl', has_title=False)

answer = []

model.eval()

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# write output to sample_submission.json, where has "title" key

with open('sample_submission.json', 'w') as f:
    for i in range(len(test_dataset)):
        if test_dataset.bodies[i] == '':
            answer.append({'title': ''})
            print(i, '')
            continue
        input = tokenizer(test_dataset.bodies[i], max_length=512, truncation=True, padding=True, return_tensors="pt")
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=150, num_beams=4, early_stopping=True)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        answer.append({'title': output})
        print(i, output)
    json.dump(answer, f)