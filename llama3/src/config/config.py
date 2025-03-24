from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 1000
    save_steps: int = 500
    
@dataclass 
class WandbConfig:
    project: str = "llama-3-bangla"
    log_model: str = "checkpoint"