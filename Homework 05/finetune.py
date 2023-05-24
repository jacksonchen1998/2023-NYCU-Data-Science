import nltk
import string
import pandas as pd
import transformers
from datasets import load_metric
from torch.utils.data import random_split
from torch.utils.data import Dataset
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

# Usage example
dataset = CustomDataset('./hw5_dataset/train.jsonl', has_title=True)
test_dataset = CustomDataset('./hw5_dataset/test.jsonl', has_title=False)

# random split train_dataset into train and validation 0.8 0.2
train_dataset, validation_dataset = random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])

print("Train dataset size: ", len(train_dataset))
print("Validation dataset size: ", len(validation_dataset))
print("Test dataset size: ", len(test_dataset))

filtered_train = [example for example in train_dataset if len(example[1]) >= 20 and len(example[0]) >= 512]
filtered_valid = [example for example in validation_dataset if len(example[1]) >= 20 and len(example[0]) >= 512]

print("Filtered train dataset size: ", len(filtered_train))
print("Filtered validation dataset size: ", len(filtered_valid))

nltk.download('punkt')

model_checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

prefix = "summarize: "
max_input_length = 512
max_target_length = 64

def clean_text(text):
    sentences = nltk.sent_tokenize(text.strip())
    sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
    sentences_cleaned_no_titles = [sent for sent in sentences_cleaned if len(sent) > 0 and sent[-1] in string.punctuation]
    text_cleaned = "\n".join(sentences_cleaned_no_titles)
    return text_cleaned

def preprocess_data(examples):
    texts_cleaned = [clean_text(text) for text in examples["text"]]
    inputs = [prefix + text for text in texts_cleaned]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["title"], max_length=max_target_length, truncation=True)
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs