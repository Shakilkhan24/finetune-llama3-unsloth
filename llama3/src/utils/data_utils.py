from datasets import load_dataset

def load_bangla_alpaca():
    """Load the Bangla Alpaca dataset"""
    return load_dataset("iamshnoo/alpaca-cleaned-bengali", split="train")

def format_alpaca_prompts(examples, prompt_template, tokenizer):
    """Format examples according to the Alpaca prompt template"""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = prompt_template.format(instruction, input_text, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts} # training data would be this format 