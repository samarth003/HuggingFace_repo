def print_sentiment(texts, results):
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} (Score: {result['score']:.4f})\n")