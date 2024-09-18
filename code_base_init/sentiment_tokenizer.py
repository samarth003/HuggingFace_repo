from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

import common_functions

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizers = AutoTokenizer.from_pretrained(model_name)

texts = ["We are very happy to show you the Huggingface Transformers library.", 
         "Nous sommes très heureux de vous présenter la bibliothèque Huggingface Transformers."]
encodings = tokenizers(text=texts)

common_functions.print_encodings(encodings)

common_functions.print_decodings(tokenizers=tokenizers, encodings=encodings)

classifier = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizers)
results = classifier(texts)
common_functions.print_sentiment(texts=texts, results=results)