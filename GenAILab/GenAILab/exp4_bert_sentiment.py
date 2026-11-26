"""
Fine-tune BERT for Sentiment Analysis on SST-2
pip install transformers datasets torch
"""
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np

# Load data & tokenizer
ds        = load_dataset("sst2")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(b):
    return tokenizer(b["sentence"], truncation=True, padding="max_length", max_length=64)

ds = ds.map(tokenize, batched=True).rename_column("label","labels")
ds.set_format("torch", columns=["input_ids","attention_mask","labels"])

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Train
trainer = Trainer(
    model=model,
    args=TrainingArguments("bert_out", num_train_epochs=1,
                           per_device_train_batch_size=16, logging_steps=50),
    train_dataset=ds["train"].select(range(500)),
    eval_dataset=ds["validation"].select(range(100)),
    compute_metrics=lambda p: {"acc": (np.argmax(p.predictions,1)==p.label_ids).mean()},
)
trainer.train()

# Inference
from transformers import pipeline
clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
print(clf("I love this!"))     # POSITIVE
print(clf("This is awful."))   # NEGATIVE
