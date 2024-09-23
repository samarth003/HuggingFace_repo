import base_model, file_common


def chatbot_app(tokenizer, model):
    print("Welcome to Chatbot service. How can I help you?")
    print("Type 'quit' to end the conversation.")
    conversation_history = []
    while True:
        user_input = input("User: ")
        if user_input == 'quit':
            print("Thank you for using this chatbot service.")
            #register delete model files call on program exit
            del_model.register_on_quit()
            break

        response = model_base.interact_with_chatbot(user_in=user_input, conv_history=conversation_history, tokenizer=tokenizer, model=model)
        print(f"Chatbot: {response}")


if __name__ == "__main__":

    model_base = base_model.base_model()
    del_model = file_common.Delete_Cached_Model()

    tokenizer, model = model_base.model_init()

    chatbot_app(tokenizer=tokenizer, model=model)