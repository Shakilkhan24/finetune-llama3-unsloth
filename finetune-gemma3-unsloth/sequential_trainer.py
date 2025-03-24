import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import torch
from torch.cuda import empty_cache
from dataclasses import dataclass
from contextlib import contextmanager

from src.config import ModelConfig, TrainingConfig, LoraConfig
from src.model import GemmaModel
from src.data_processing import DataProcessor
from src.trainer import GemmaTrainer

@dataclass
class TrainingResult:
    iteration: int
    model_dir: str
    training_loss: float
    timestamp: str
    metrics: Dict

class SequentialTrainer:
    def __init__(self, base_output_dir: str = "models/iterations"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log = []
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Initialize logging configuration"""
        self.log_file = self.base_output_dir / "training.log"
        self.progress_file = self.base_output_dir / "training_progress.json"

    @contextmanager
    def _cuda_memory_manager(self):
        """Context manager for CUDA memory management"""
        try:
            yield
        finally:
            empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def _get_iteration_dir(self, iteration: int) -> Path:
        """Create and return directory for this iteration's outputs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        iteration_dir = self.base_output_dir / f"iteration_{iteration}_{timestamp}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        return iteration_dir

    def _save_metrics(self, metrics: Dict, iteration_dir: Path) -> None:
        """Save training metrics to JSON"""
        metrics_file = iteration_dir / "training_metrics.json"
        metrics["timestamp"] = datetime.now().isoformat()
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        self.metrics_log.append(metrics)

    def _load_previous_model(self, iteration: int) -> Optional[str]:
        """Load the most recent model from previous iteration"""
        if iteration <= 1:
            return None
        
        prev_dirs = list(self.base_output_dir.glob(f"iteration_{iteration-1}_*"))
        if not prev_dirs:
            return None
        
        latest_dir = max(prev_dirs, key=lambda x: x.stat().st_mtime)
        model_path = latest_dir / "model"
        return str(model_path) if model_path.exists() else None

    @staticmethod
    def get_dataset_for_iteration(iteration: int) -> str:
        """Return dataset based on iteration number"""
        datasets = {
            1: "mlabonne/FineTome-100k",      # Financial domain
            2: "sahil2801/CodeAlpaca-20k",    # Coding domain
            3: "databricks/databricks-dolly-15k"  # General knowledge
        }
        return datasets.get(iteration, datasets[1])  # Default to first dataset

    def train_iteration(self, iteration: int, dataset_name: Optional[str] = None) -> TrainingResult:
        """Run a single training iteration"""
        iteration_dir = self._get_iteration_dir(iteration)
        
        with self._cuda_memory_manager():
            try:
                # Model initialization
                gemma = GemmaModel(ModelConfig)
                prev_model_path = self._load_previous_model(iteration)
                model, tokenizer = gemma.setup_model(model_path=prev_model_path)

                # Dataset processing
                data_processor = DataProcessor(tokenizer)
                dataset_name = dataset_name or self.get_dataset_for_iteration(iteration)
                dataset = data_processor.load_and_process_dataset(dataset_name)

                # Training
                trainer = GemmaTrainer(model, tokenizer, dataset, TrainingConfig)
                sft_trainer = trainer.setup_trainer()
                training_stats = trainer.train(sft_trainer)

                # Save outputs
                model_dir = iteration_dir / "model"
                trainer.save_model(output_dir=str(model_dir))
                self._save_metrics(training_stats, iteration_dir)

                return TrainingResult(
                    iteration=iteration,
                    model_dir=str(model_dir),
                    training_loss=training_stats.get("training_loss", 0.0),
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    metrics=training_stats
                )

            except Exception as e:
                self._log_error(iteration, str(e))
                raise

    def run_sequential_training(
        self, 
        num_iterations: int = 10, 
        datasets: Optional[List[str]] = None
    ) -> List[TrainingResult]:
        """Run multiple training iterations"""
        results = []

        for i in range(num_iterations):
            try:
                dataset_name = datasets[i] if datasets else None
                print(f"\n{'='*50}")
                print(f"Starting Training Round {i+1}/{num_iterations}")
                print(f"{'='*50}")

                result = self.train_iteration(i + 1, dataset_name)
                results.append(result)

                # Save progress
                self._save_progress(results)

            except Exception as e:
                print(f"Critical error in iteration {i+1}: {str(e)}")
                self._save_progress(results)  # Save progress before stopping
                break

        return results

    def evaluate_model(
        self, 
        model_dir: str, 
        test_questions: List[str],
        max_new_tokens: int = 64,
        temperature: float = 0.7
    ) -> List[Dict]:
        """Evaluate model with test questions"""
        with self._cuda_memory_manager():
            gemma = GemmaModel(ModelConfig)
            model, tokenizer = gemma.setup_model()
            
            # Load fine-tuned weights
            model_path = Path(model_dir) / "pytorch_model.bin"
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set to evaluation mode

            results = []
            for question in test_questions:
                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": question}]
                }]

                text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                )

                with torch.no_grad():
                    outputs = model.generate(
                        **tokenizer([text], return_tensors="pt").to("cuda"),
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=0.95,
                        top_k=64,
                        do_sample=True
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append({"question": question, "response": response})

            return results

    def _save_progress(self, results: List[TrainingResult]) -> None:
        """Save training progress to JSON"""
        with open(self.progress_file, "w") as f:
            json.dump([vars(r) for r in results], f, indent=4)

    def _log_error(self, iteration: int, error_msg: str) -> None:
        """Log error messages"""
        with open(self.log_file, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] Error in iteration {iteration}: {error_msg}\n")