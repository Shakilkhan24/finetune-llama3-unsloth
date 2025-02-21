# Llama 3 Bengali Fine-tuning Project

Fine-tune Llama 3 for the Bengali language using the Unsloth framework, with support for GGUF export and Ollama deployment.

## 📁 Project Structure
```
.
├── scripts/
│   ├── train.py           # Main training script
│   ├── save_gguf.py       # Script to save model in GGUF format
│   └── setup_ollama.sh    # Shell script for Ollama setup and deployment
├── src/
│   ├── config/            # Configuration files
│   ├── model/             # Model training and inference code
│   └── utils/             # Utility functions for data processing
└── README.md              # This file
```

## 🚀 Setup
1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have sufficient disk space for model weights and GGUF formats.

## 🎯 Training
1. Configure training parameters in `src/config/config.py`
2. Run the training script:
   ```bash
   python scripts/train.py
   ```
   This will:
   - Fine-tune the Llama 3 model on Custom data
   - Save checkpoints during training
   - Upload the model to Hugging Face Hub

## GGUF Export and Ollama Deployment
### Save the model in GGUF format
```bash
python scripts/save_gguf.py
```
This creates:
- **Q8_0**: 8-bit quantized version (higher quality, larger size)
- **Q4_K_M**: 4-bit quantized version (smaller, slightly lower quality)
- Ollama Modelfile

### Deploy to Ollama
```bash
chmod +x scripts/setup_ollama.sh
./scripts/setup_ollama.sh
```
This will:
- Install Ollama if not present
- Start the Ollama server
- Create the model in Ollama
- Run a test query

## Using the Model
After deployment, you can interact with the model using:

### 🔹 Command Line
```bash
ollama run bangla-llama "Your prompt here"
```

### 🔹 API
```bash
curl http://localhost:11434/api/chat -d '{
    "model": "bangla-llama",
    "messages": [
        {
            "role": "user",
            "content": "Your prompt here"
        }
    ]
}'
```

## 📦 Model Versions
- **Q8_0**: 8-bit quantized version (better quality, requires more resources)
- **Q4_K_M**: 4-bit quantized version (smaller, more memory-efficient)
