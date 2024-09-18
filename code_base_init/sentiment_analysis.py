from transformers import pipeline
# import torch

import common_functions

sentiment_analysis = pipeline("sentiment-analysis")

texts = ["I am struggling to get into AI-ML stream.", 
         "Pasta served today at lunch was cold"]

results = sentiment_analysis(texts)

common_functions.print_sentiment(texts=texts, results=results)