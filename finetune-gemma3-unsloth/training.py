from src.config import ModelConfig, TrainingConfig, LoraConfig
from src.model import GemmaModel
from src.data_processing import DataProcessor
from src.trainer import GemmaTrainer

# Initialize model
gemma = GemmaModel(ModelConfig)
model, tokenizer = gemma.setup_model()

# Process dataset
data_processor = DataProcessor(tokenizer)
dataset = data_processor.load_and_process_dataset("mlabonne/FineTome-100k")

# Setup trainer
trainer = GemmaTrainer(model, tokenizer, dataset, TrainingConfig)
sft_trainer = trainer.setup_trainer()

# Train model
training_stats = trainer.train(sft_trainer)

# Save model
trainer.save_model()

# Test the model
messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "What is Gemma-3?"}]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer([text], return_tensors="pt").to("cuda"),
    max_new_tokens=64,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)