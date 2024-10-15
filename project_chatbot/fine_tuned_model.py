from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments
from transformers import BatchEncoding
import numpy as np

from base_model import base_model

DATASET_NAME = "allenai/WildChat"

class peft_model(base_model):

    def __init__(self):
        super().__init__()
        self.model_base = None
        self.tokenizer_base = None
        self.wildchat_train_db = None
        self.wildchat_test_db = None

    def import_dataset(self):
        dataset = load_dataset(DATASET_NAME)
        # print(dataset["train"][0])
        return dataset

    def model_config(self):
        self.tokenizer_base, self.model_base = self.model_init()
        # print(model_base)
        # peft_cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        # model_peft = get_peft_model(model=model_base, peft_config=peft_cfg)
        # model_peft.print_trainable_parameters()

        # def tokenize_in(examples):
        #     if "conversation" in examples:
        #         conversation_data = examples["conversation"]

        #         # Print out the structure of conversation_data
        #         # print(f"Conversation data: {conversation_data}")

        #         if isinstance(conversation_data, list):
        #             conversations = []

        #             # Loop through each turn in the conversation
        #             for turn_list in conversation_data:
        #                 if isinstance(turn_list, list):
        #                     # Loop through each turn within the list
        #                     for turn in turn_list:
        #                         if isinstance(turn, dict) and "content" in turn:
        #                             # Extract the content if it exists
        #                             conversations.append(turn["content"])
        #                         else:
        #                             # Handle unexpected structures
        #                             print(f"Unexpected format in conversation turn: {turn}")
        #                 elif isinstance(turn_list, dict) and "content" in turn_list:
        #                     # Handle case where conversation_data is a list of dicts directly
        #                     conversations.append(turn_list["content"])
        #                 else:
        #                     print(f"Unexpected format in conversation turn list: {turn_list}")

        #             # Check if conversations list is empty before tokenizing
        #             if not conversations:
        #                 print(f"No valid content found in the examples: {examples}")
        #                 return {}  # Return None or some default value to prevent an error

        #             # Tokenize the conversation texts
        #             return self.tokenizer_base(conversations, padding="max_length", truncation=True, max_length=128)
        #         else:
        #             print(f"Unexpected type for 'conversation': {type(conversation_data)}")
        #     else:
        #         print(f"'conversation' key not found in examples: {examples}")

        #     return {}

        def tokenize_in(examples):
            if "conversation" in examples:
                conversation_data = examples["conversation"]

                if isinstance(conversation_data, list):
                    conversations = []

                    # Extract content from nested lists of dictionaries
                    for turn_list in conversation_data:
                        if isinstance(turn_list, list):
                            for turn in turn_list:
                                if isinstance(turn, dict) and "content" in turn:
                                    # print(turn["content"])
                                    conversations.append(turn["content"])

                    if not conversations:
                        print(f"No valid content found in the examples: {examples}")
                        return {"input_ids": [], "attention_mask": []}

                    # Tokenize with fixed max_length to ensure consistent output length
                    max_length = 32  # Set the maximum length you want for padding/truncation
                    tokenized_output = self.tokenizer_base(
                        conversations,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="np"  # Use Numpy for easier post-processing
                    )

                    # Ensure that input_ids and attention_mask are consistently padded
                    input_ids = tokenized_output["input_ids"]
                    attention_mask = tokenized_output["attention_mask"]

                    # Convert to lists and make sure they are of consistent length
                    input_ids = input_ids.tolist()
                    attention_mask = attention_mask.tolist()

                    # for ids in input_ids:
                    #     print(len(ids))

                    # Debug: Check that all input_ids are of the same length (max_length)
                    if not all(len(ids) == max_length for ids in input_ids):
                        print(f"Inconsistent lengths detected in input_ids: {input_ids}")
                        return {"input_ids": [], "attention_mask": []}

                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask
                    }
                else:
                    print(f"Unexpected type for 'conversation': {type(conversation_data)}")
            else:
                print(f"'conversation' key not found in examples: {examples}")

            return {"input_ids": [], "attention_mask": []}

        # Use map with smaller batch size to ensure stability
        db = self.import_dataset()
        try:
            db = db.remove_columns(column_names=['conversation_id', 'model', 'timestamp', 'turn', 
                                                 'language', 'openai_moderation', 'detoxify_moderation', 
                                                 'toxic', 'redacted'
                                                 ])
            wildchat_db = db.map(tokenize_in, 
                                 batched=True, 
                                 batch_size=32,
                                )

        except Exception as e:
            print(f"Error during babtch processing: {e}")
            raise
        self.wildchat_train_db = wildchat_db["train"]
        self.wildchat_test_db = wildchat_db["test"]

    def model_train(self):
        training_args = TrainingArguments(output_dir="test_trainer/chatbot_ft", 
                                          learning_rate=1e-3,
                                          eval_strategy="epoch", 
                                          save_strategy="epoch", 
                                          load_best_model_at_end=True
                                          )
        self.model_trainer = Trainer(model=self.model_base,
                                args=training_args, 
                                train_dataset=self.wildchat_train_db,
                                eval_dataset=self.wildchat_test_db,
                                )
        self.model_trainer.train()
    
    def model_save(self):
        self.model_trainer.save_model(output_dir="test_trainer/chatbot_ft/fine_tuned_model")

    def model_load(self):
        pass
        


if __name__ == "__main__":
    peft_m = peft_model()
    # peft_m.import_dataset()
    peft_m.model_config()
    peft_m.model_train()
    peft_m.model_save()
    