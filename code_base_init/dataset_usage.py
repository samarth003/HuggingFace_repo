from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import pipeline
import numpy as np
import evaluate

import common_functions

dataset = load_dataset("Yelp/yelp_review_full")

# model_name = "SamLowe/roberta-base-go_emotions"
model_name = "google-bert/bert-base-cased"

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5, ignore_mismatched_sizes=True)
tokenizers = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizers(examples["text"], padding="max_length", truncation=True)

#Tokenization
tokenized_db = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_db["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_db["test"].shuffle(seed=42).select(range(1000))

# print(small_train_dataset[100:105])

#Train
training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

#Evaluate
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# texts = dataset["train"][100:104]['text']

# classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizers)
# results = classifier(texts)
# common_functions.print_sentiment(texts=texts, results=results)