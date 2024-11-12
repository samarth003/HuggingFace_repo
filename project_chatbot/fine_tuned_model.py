from datasets import load_dataset, load_from_disk
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
        def tokenize_in(examples):
            if "conversation" in examples:
                conversation_data = examples["conversation"]

                if isinstance(conversation_data, list):
                    conversations = []
                    for turn_list in conversation_data:
                        if isinstance(turn_list, list):
                            turns_content = " ".join(turn.get("content", "") for turn in turn_list if isinstance(turn, dict))
                            conversations.append(turns_content)

                    if not conversations:
                        print(f"No valid content found in the examples: {examples}")
                        return {"input_ids": [], "attention_mask": []}

                    # Tokenize with fixed max_length to ensure consistent output length
                    max_length = 32  # Set the maximum length you want for padding/truncation
                    tokenized_output = self.tokenizer_base(
                        conversations,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length
                    )

                    # Ensure that input_ids and attention_mask are consistently padded
                    input_ids = tokenized_output["input_ids"]
                    attention_mask = tokenized_output["attention_mask"]

                    return tokenized_output
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
            self.wildchat_db = db.map(tokenize_in, 
                                 batched=True, 
                                 batch_size=32,
                                )

        except Exception as e:
            print(f"Error during batch processing: {e}")
            raise
        #test train split required
        self.wildchat_db.save_to_disk("test_trainer/mapped_db")

    def model_train(self):
        self.wildchat_db = load_from_disk("test_trainer/mapped_db") 
        wildchat_db_split = self.wildchat_db.train_test_split(test_size=0.2, shuffle=True)
        self.wildchat_train_db = wildchat_db_split["train"]
        self.wildchat_test_db = wildchat_db_split["test"]
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
    