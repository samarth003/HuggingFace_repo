from datasets import load_dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments
from transformers import BatchEncoding
import numpy as np

from base_model import base_model

DATASET_NAME = "allenai/WildChat"
DATABASE_MAPPED = True

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

    def model_tokenize(self):
        self.tokenizer_base, self.model_base = self.model_init()
        if DATABASE_MAPPED != True:
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
                            return {"input_ids": [], "attention_mask": [], 
                                    "decoder_input_ids": [], "labels": []}

                        # Tokenize with fixed max_length to ensure consistent output length
                        max_length = 32  # Set the maximum length you want for padding/truncation
                        tokenize_examples = self.tokenizer_base(
                            conversations,
                            padding="max_length",
                            truncation=True,
                            max_length=max_length
                        )
                        input_ids = tokenize_examples["input_ids"]
                        attention_mask = tokenize_examples["attention_mask"]
                        decoder_input_ids = [
                            [self.tokenizer_base.pad_token_id] + ids[:-1] for ids in input_ids
                        ]
                        labels = input_ids.copy()

                        # return tokenize_examples
                        return {
                            "input_ids"        : input_ids,
                            "attention_mask"   : attention_mask,
                            "decoder_input_ids": decoder_input_ids,
                            "labels"           : labels
                        }
                    else:
                        print(f"Unexpected type for 'conversation': {type(conversation_data)}")
                else:
                    print(f"'conversation' key not found in examples: {examples}")

                return {"input_ids": [], "attention_mask": [], "decoder_input_ids": [], "labels": []}

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
        else:
            pass
        
    def model_train(self):
        self.wildchat_db = load_from_disk("test_trainer/mapped_db") 
        wildchat_db_split = self.wildchat_db["train"].train_test_split(test_size=0.2, shuffle=True)
        wildchat_split_db = {
            "train": wildchat_db_split["train"],
            "test" : wildchat_db_split["test"]
        }

        self.wildchat_train_db = wildchat_split_db["train"]
        self.wildchat_test_db = wildchat_split_db["test"]

        training_args = TrainingArguments(output_dir="test_trainer/chatbot_ft", 
                                          learning_rate=1e-4,
                                          eval_strategy="no", 
                                          save_strategy="no", 
                                          load_best_model_at_end=True,
                                          per_device_eval_batch_size=8,
                                          per_device_train_batch_size=8,
                                          gradient_accumulation_steps=2,
                                          num_train_epochs=2
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
    peft_m.model_tokenize()
    peft_m.model_train()
    peft_m.model_save()
    