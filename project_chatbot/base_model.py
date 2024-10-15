from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

MODEL_NAME = "facebook/blenderbot-400M-distill"

class base_model():

    def __init__(self):
        self.model_hf = MODEL_NAME

    def model_init(self, model_name = MODEL_NAME):
        #load pre-trained model and tokenizer
        model_name = self.model_hf
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model

    #Funtion to interact with chatbot 
    def interact_with_chatbot(self, user_in, conv_history, tokenizer, model):
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
    
if __name__ == "__main__":

    user_input = "Tell me a joke"
    conversation_history = []
    model_b = base_model()
    tokenizer, model = model_b.model_init()
    response = model_b.interact_with_chatbot(user_in=user_input, conv_history=conversation_history, tokenizer=tokenizer, model=model)
    print(f"Chatbot: {response}")