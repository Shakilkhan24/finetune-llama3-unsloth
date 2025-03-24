import argparse
from pathlib import Path
from datetime import datetime
from src.config import ModelConfig, TrainingConfig, LoraConfig
from src.model import GemmaModel
from src.data_processing import DataProcessor
from src.trainer import GemmaTrainer

def setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Fine-tune Gemma model on a specific dataset')
    
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to existing model directory. If not provided, will use base Gemma model'
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to the local dataset directory or Hugging Face dataset name'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models/finetuned',
        help='Directory to save the fine-tuned model'
    )
    
    parser.add_argument(
        '--test_prompt',
        type=str,
        default='What is machine learning?',
        help='Test prompt to evaluate the model after training'
    )
    
    return parser

def save_training_info(output_dir: Path, model_path: str, dataset_path: str, training_stats: dict):
    """Save training metadata and stats"""
    info = {
        'timestamp': datetime.now().isoformat(),
        'base_model': str(model_path) if model_path else 'base Gemma',
        'dataset': dataset_path,
        'training_stats': training_stats
    }
    
    info_file = output_dir / 'training_info.json'
    import json
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=4)

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"finetuned_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    print(f"\nInitializing model...")
    gemma = GemmaModel(ModelConfig)
    model, tokenizer = gemma.setup_model(model_path=args.model_path)
    
    # Process dataset
    print(f"\nLoading and processing dataset: {args.dataset_path}")
    data_processor = DataProcessor(tokenizer)
    dataset = data_processor.load_and_process_dataset(args.dataset_path)
    
    # Setup trainer
    print("\nSetting up trainer...")
    trainer = GemmaTrainer(model, tokenizer, dataset, TrainingConfig)
    sft_trainer = trainer.setup_trainer()
    
    # Train model
    print("\nStarting training...")
    training_stats = trainer.train(sft_trainer)
    
    # Save model and training info
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir=str(output_dir))
    save_training_info(output_dir, args.model_path, args.dataset_path, training_stats)
    
    # Test the model
    print("\nTesting model with prompt:", args.test_prompt)
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": args.test_prompt}]
    }]
    
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )
    
    from transformers import TextStreamer
    streamer = TextStreamer(tokenizer)
    
    print("\nModel response:")
    _ = model.generate(
        **tokenizer([text], return_tensors="pt").to("cuda"),
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        streamer=streamer
    )
    print("\nDone!")

if __name__ == "__main__":
    main()