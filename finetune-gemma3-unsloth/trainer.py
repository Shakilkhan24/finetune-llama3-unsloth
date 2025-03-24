from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
import torch

class GemmaTrainer:
    def __init__(self, model, tokenizer, dataset, config):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config
        
    def setup_trainer(self):
        training_config = SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=self.config.WARMUP_STEPS,
            max_steps=self.config.MAX_STEPS,
            learning_rate=self.config.LEARNING_RATE,
            logging_steps=self.config.LOGGING_STEPS,
            optim="adamw_8bit",
            weight_decay=self.config.WEIGHT_DECAY,
            lr_scheduler_type="linear",
            seed=self.config.SEED,
            report_to="none",
        )
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            args=training_config,
        )
        
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )
        
        return trainer
    
    def train(self, trainer):
        stats = trainer.train()
        return stats
    
    def save_model(self, output_dir="models/checkpoints"):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)