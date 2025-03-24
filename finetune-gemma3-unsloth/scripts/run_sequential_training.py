from src.sequential_trainer import SequentialTrainer
from typing import List, Dict

def evaluate_on_all_domains(trainer: SequentialTrainer, model_dir: str, domain_questions: Dict[str, List[str]]) -> None:
    """Evaluate the model on questions from all domains"""
    print("\nEvaluating model on all domains:")
    print("="*50)
    for domain, questions in domain_questions.items():
        print(f"\n{domain} Domain Evaluation:")
        print("-"*30)
        eval_results = trainer.evaluate_model(
            model_dir,
            questions,
            max_new_tokens=64,
            temperature=0.7
        )
        for eval_result in eval_results:
            print(f"\nQ: {eval_result['question']}")
            print(f"A: {eval_result['response']}")
        print("-"*50)

def main():
    # Initialize the sequential trainer
    trainer = SequentialTrainer(base_output_dir="models/sequential_training")
    
    # Define test questions for each domain
    domain_questions = {
        "Finance": [
            "What is a stock market index?",
            "Explain the concept of compound interest.",
            "What is the difference between stocks and bonds?",
        ],
        "Programming": [
            "What is object-oriented programming?",
            "Explain the concept of recursion.",
            "What is the difference between Python lists and tuples?",
        ],
        "General Knowledge": [
            "What is machine learning?",
            "Explain the concept of climate change.",
            "What is the scientific method?",
        ]
    }
    
    # Define the sequence of datasets for fine-tuning
    dataset_sequence = [
        "mlabonne/FineTome-100k",      # First: Financial domain
        "sahil2801/CodeAlpaca-20k",    # Second: Programming domain
        "databricks/databricks-dolly-15k"  # Third: General knowledge
    ]
    
    # Run the sequential training
    results = trainer.run_sequential_training(
        num_iterations=len(dataset_sequence),
        datasets=dataset_sequence
    )
    
    # Print summary of all iterations
    print("\nTraining Summary:")
    print("="*50)
    for result in results:
        print(f"\nIteration {result.iteration}:")
        print(f"Model saved at: {result.model_dir}")
        print(f"Training loss: {result.training_loss}")
        print(f"Completed at: {result.timestamp}")
        
        # Evaluate the model from this iteration on all domains
        evaluate_on_all_domains(trainer, result.model_dir, domain_questions)

if __name__ == "__main__":
    main()