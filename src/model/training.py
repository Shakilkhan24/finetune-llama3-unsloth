from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
import wandb
import torch

class LlamaBanglaTrainer:
    def __init__(self, 
                 model_name="unsloth/llama-3-8b-bnb-4bit",
                 max_seq_length=2048,
                 load_in_4bit=True):
        self.model, self.tokenizer = self._load_base_model(
            model_name, max_seq_length, load_in_4bit
        )
        
    def _load_base_model(self, model_name, max_seq_length, load_in_4bit):
        return FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            device_map='auto'
        )

    def setup_lora(self, r=16):
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    def train(self, 
              dataset,
              output_dir,
              run_name,
              max_steps=1000,
              save_steps=500,
              resume_from_checkpoint=False,
              **kwargs):
        
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            save_steps=save_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            report_to="wandb",
            run_name=run_name,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            **kwargs
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )

        trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return trainer_stats 