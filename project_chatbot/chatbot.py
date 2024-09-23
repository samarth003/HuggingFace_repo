import os
import atexit
import shutil
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

MODEL_NAME = "facebook/blenderbot-400M-distill"

def model_init(model_name = MODEL_NAME):
    #load pre-trained model and tokenizer
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

#Funtion to interact with chatbot 
def interact_with_chatbot(user_in, conv_history, tokenizer, model):
    #add user inputs to conversation history
    conv_history.append(f"User: {user_in}")
    #prepare the input text for the model, use only 
    #last 5 exchanges for precise context
    conv_text = " ".join(conv_history[-5:])
    #generate a response using chatbot pipeline 
    inputs = tokenizer(conv_text, return_tensors="pt", truncation=True)
    response_ids = model.generate(**inputs, max_length=1000, pad_token_id = tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    #add response to conversation history
    conv_history.append(f"Chatbot: {response_text}")
    return response_text

def delete_model_files():
    cache_dir = os.path.expanduser("C:\Users\samar\.cache\huggingface\hub\models--facebook--blenderbot-400M-distill")
    if os.path.exists(cache_dir):
        user_in = input("Do you want to delete the model files stored in cache dir? (y/n): ")
        if user_in.lower() == 'y':
            shutil.rmtree(cache_dir)
            print("Deleted model files from cached directory!")
        else:
            print("Model files not deleted.")
    else:
        print(f"Model files not found in path: {cache_dir}")

#start conversation loop 
if __name__ == "__main__":

    tokenizer, model = model_init(model_name=MODEL_NAME)

    print("Welcome to Chatbot service. How can I help you?")
    print("Type 'quit' to end the conversation.")
    conversation_history = []
    while True:
        user_input = input("User: ")
        if user_input == 'quit':
            print("Thank you for using this chatbot service.")
            #register delete model files call on program exit
            atexit.register(delete_model_files)
            break

        response = interact_with_chatbot(user_in=user_input, conv_history=conversation_history, tokenizer=tokenizer, model=model)
        print(f"Chatbot: {response}")