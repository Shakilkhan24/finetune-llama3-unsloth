import os
from src.model.training import LlamaBanglaTrainer
from src.config.config import TrainingConfig

def save_model_gguf():
    # Initialize trainer with your trained model
    trainer = LlamaBanglaTrainer(
        model_name="Shakil2448868/llama-3-bangla-lora-sample",  # [NOTE] Your trained model
        max_seq_length=TrainingConfig.max_seq_length,
        load_in_4bit=TrainingConfig.load_in_4bit
    )
    
    # Save model in different GGUF formats
    save_path = "gguf_models"
    os.makedirs(save_path, exist_ok=True)
    
    # Save Q8_0 version (8-bit)
    trainer.model.save_pretrained_gguf(
        f"{save_path}/q8_0",
        trainer.tokenizer,
        quantization_method="q8_0"
    )
    
    # Save Q4_K_M version (4-bit)
    trainer.model.save_pretrained_gguf(
        f"{save_path}/q4_k_m",
        trainer.tokenizer,
        quantization_method="q4_k_m"
    )


if __name__ == "__main__":
    save_model_gguf() 