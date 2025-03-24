from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import torch
from pathlib import Path
from typing import Optional, Tuple

class GemmaModel:
    def __init__(self, config):
        self.config = config
        
    def setup_model(self, model_path: Optional[str] = None) -> Tuple[FastModel, any]:
        """Setup the model, optionally loading from a saved checkpoint
        
        Args:
            model_path: Optional path to a directory containing a saved model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_path:
            print(f"Loading model from: {model_path}")
            model_path = Path(model_path)
            if not model_path.exists():
                raise ValueError(f"Model path does not exist: {model_path}")
                
            # Load the base model first
            model, tokenizer = FastModel.from_pretrained(
                model_name=self.config.MODEL_NAME,
                max_seq_length=self.config.MAX_SEQ_LENGTH,
                load_in_4bit=self.config.LOAD_IN_4BIT,
                load_in_8bit=self.config.LOAD_IN_8BIT,
                full_finetuning=self.config.FULL_FINETUNING,
            )
            
            # Setup PEFT configuration
            model = FastModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=self.config.R,
                lora_alpha=self.config.LORA_ALPHA,
                lora_dropout=self.config.LORA_DROPOUT,
                bias=self.config.BIAS,
                random_state=self.config.SEED,
            )
            
            # Load the fine-tuned weights
            state_dict_path = model_path / "pytorch_model.bin"
            if not state_dict_path.exists():
                raise ValueError(f"Model weights not found at: {state_dict_path}")
            model.load_state_dict(torch.load(state_dict_path))
            
        else:
            print("Loading base Gemma model...")
            model, tokenizer = FastModel.from_pretrained(
                model_name=self.config.MODEL_NAME,
                max_seq_length=self.config.MAX_SEQ_LENGTH,
                load_in_4bit=self.config.LOAD_IN_4BIT,
                load_in_8bit=self.config.LOAD_IN_8BIT,
                full_finetuning=self.config.FULL_FINETUNING,
            )
            
            model = FastModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=self.config.R,
                lora_alpha=self.config.LORA_ALPHA,
                lora_dropout=self.config.LORA_DROPOUT,
                bias=self.config.BIAS,
                random_state=self.config.SEED,
            )
        
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-3",
        )
        
        return model, tokenizer