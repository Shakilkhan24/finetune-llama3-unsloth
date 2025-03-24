from datasets import load_dataset
from unsloth.chat_templates import standardize_data_formats

class DataProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def load_and_process_dataset(self, dataset_name, split="train"):
        dataset = load_dataset(dataset_name, split=split)
        dataset = standardize_data_formats(dataset)
        
        def apply_chat_template(examples):
            texts = self.tokenizer.apply_chat_template(examples["conversations"])
            return {"text": texts}
        
        dataset = dataset.map(apply_chat_template, batched=True)
        return dataset