from unsloth import FastLanguageModel
from transformers import TextStreamer, AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

class LlamaBanglaInference:
    def __init__(self, model_path, use_unsloth=True, load_in_4bit=True):
        self.model_path = model_path
        self.load_in_4bit = load_in_4bit
        
        if use_unsloth:
            self._load_unsloth_model()
        else:
            self._load_huggingface_model()

    def _load_unsloth_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)

    def _load_huggingface_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def generate(self, instruction, input_text="", max_new_tokens=1024, stream=False):
        prompt = self._format_prompt(instruction, input_text)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        if stream:
            streamer = TextStreamer(self.tokenizer)
            outputs = self.model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id
            )
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
            return self.tokenizer.batch_decode(outputs)

    def _format_prompt(self, instruction, input_text):
        return f"""Below is an instruction {instruction} in bangla that describes a task, paired with an input also in bangla that provides further context : {input_text} . Write a response in bangla that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
""" 