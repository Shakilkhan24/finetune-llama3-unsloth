class ModelConfig:
    MODEL_NAME = "unsloth/gemma-3-4b-it"
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True
    LOAD_IN_8BIT = False
    FULL_FINETUNING = False

class TrainingConfig:
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    WARMUP_STEPS = 5
    MAX_STEPS = 30
    LEARNING_RATE = 2e-4
    LOGGING_STEPS = 1
    WEIGHT_DECAY = 0.01
    SEED = 3407
    
class LoraConfig:
    R = 8
    LORA_ALPHA = 8
    LORA_DROPOUT = 0
    BIAS = "none"