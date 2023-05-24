import pandas as pd
import numpy as np
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import tensorboard
from transformers import AutoTokenizer
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
import nltk
import string
import evaluate

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
train_dataset = CustomDataset('./hw5_dataset/train.jsonl', has_title=True)
test_dataset = CustomDataset('./hw5_dataset/test.jsonl', has_title=False)

news = {'train': Dataset.from_dict({'body': train_dataset.bodies, 'title': train_dataset.titles}),
        'test': Dataset.from_dict({'body': test_dataset.bodies}),
}

checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + body for body in examples["body"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["title"], max_length=150, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# randpm split

news_clean_dataset = news['train'].filter(
    lambda example: (len(example['body']) >= 500) and
    (len(example['title']) >= 20)
) # 82106

news_train_validation = news_clean_dataset.train_test_split(test_size=0.2, seed=42)
news_train_dataset = news_train_validation['train'] # 65684
news_validation_dataset = news_train_validation['test'] # 16422

# print(len(news_train_dataset))
# print(len(news_validation_dataset))

news_train_dataset = news_train_dataset.shuffle(seed=42).select(range(len(news_train_dataset)))
news_validation_dataset = news_validation_dataset.shuffle(seed=42).select(range(len(news_validation_dataset)))

nltk.download('punkt')

model_checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

prefix = "summarize: "
max_input_length = 512
max_target_length = 64

def clean_text(text):
    sentences = nltk.sent_tokenize(text.strip())
    sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
    sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                    if len(sent) > 0 and
                                    sent[-1] in string.punctuation]
    text_cleaned = "\n".join(sentences_cleaned_no_titles)
    return text_cleaned

def preprocess_data(examples):
    texts_cleaned = [clean_text(text) for text in examples["body"]]
    inputs = [prefix + text for text in texts_cleaned]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["title"], max_length=max_target_length, 
                        truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = news_train_dataset.map(preprocess_data, batched=True)
validation_dataset = news_validation_dataset.map(preprocess_data, batched=True)

batch_size = 8
model_name = "t5-small-finetuned-summarizer"
model_dir = "./model/"

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_checkpoint)
metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                      for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

trainer = Seq2SeqTrainer(
    model=model_init(),
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()