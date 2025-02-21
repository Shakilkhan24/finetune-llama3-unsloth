#!/bin/bash

# Install Ollama if not already installed
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama server
ollama serve &

# Wait for server to start
sleep 5

# Create Ollama model using the saved GGUF model
ollama create bangla-llama -f gguf_models/Modelfile

# Test the model with a sample query
curl http://localhost:11434/api/chat -d '{
    "model": "bangla-llama",
    "messages": [
        {
            "role": "user",
            "content": "আপনি কেমন আছেন?"
        }
    ]
}'

echo "\nModel is ready to use!" 