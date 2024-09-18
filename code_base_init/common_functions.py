def print_sentiment(texts, results):
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} (Score: {result['score']:.4f})\n")

def print_encodings(encodings):
    print(encodings)

def print_decodings(tokenizers, encodings):
    list_len = len(encodings["input_ids"])
    print(list_len)
    for i in range(0,list_len):
        print(tokenizers.decode(encodings["input_ids"][i]))
        