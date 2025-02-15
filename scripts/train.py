import os

import wandb
from src.model.training import LlamaBanglaTrainer
from src.utils.data_utils import load_bangla_alpaca, format_alpaca_prompts
from src.utils.prompt_utils import ALPACA_PROMPT_TEMPLATE
from src.config.config import TrainingConfig, WandbConfig

def main():
    # Setup wandb
    os.environ["WANDB_PROJECT"] = WandbConfig.project
    os.environ["WANDB_LOG_MODEL"] = WandbConfig.log_model
    wandb.login()

    # Initialize trainer with your previously fine-tuned model
    trainer = LlamaBanglaTrainer(
        model_name="unsloth/llama-3-8b-bnb-4bit",  # Your uploaded model
        max_seq_length=TrainingConfig.max_seq_length,
        load_in_4bit=TrainingConfig.load_in_4bit
    )
    
    # Setup LoRA
    trainer.setup_lora(r=TrainingConfig.lora_r)
    
    # Load and format dataset
    dataset = load_bangla_alpaca()
    formatted_dataset = dataset.map(
        lambda x: format_alpaca_prompts(x, ALPACA_PROMPT_TEMPLATE, trainer.tokenizer),
        batched=True
    )
    
    # Train
    trainer.train(
        dataset=formatted_dataset,
        output_dir="outputs",
        run_name="initial_training",
        max_steps=TrainingConfig.max_steps,
        save_steps=TrainingConfig.save_steps
    )

    # After training, upload to HuggingFace
    trainer.model.push_to_hub("Shakil2448868/llama-3-bangla-lora-sample", token="your_hf_token")
    trainer.tokenizer.push_to_hub("Shakil2448868/llama-3-bangla-lora-sample", token="your_hf_token")

    # For 4-bit quantized version:
    trainer.model.push_to_hub_merged(
        "Shakil2448868/llama-3-bangla-4bit-sample",
        trainer.tokenizer,
        save_method="merged_4bit_forced",
        token="your_hf_token"
    )

if __name__ == "__main__":
    main() 