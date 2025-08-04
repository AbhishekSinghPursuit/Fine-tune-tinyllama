# Fine-tuned TinyLlama for JSON Extraction

A fine-tuned language model for extracting structured product information from HTML snippets and outputting it in JSON format.

## Download from Hugging Face

- Fine-tuned Model Repository: [learn-abc/html-model-tinyllama-chat-bnb-4bit](https://huggingface.co/learn-abc/html-model-tinyllama-chat-bnb-4bit)
- GGUF Quantized Model Repository: [learn-abc/html-model-tinyllama-chat-bnb-4bit-gguf](https://huggingface.co/learn-abc/html-model-tinyllama-chat-bnb-4bit-gguf)

## Setup

This section outlines how to set up the project environment.

### 1. Installation of required libraries.

Install required packages:
```bash
!pip install unsloth trl peft accelerate bitsandbytes huggingface_hub -q
```

### 2. Prerequisites (e.g., Python environment, GPU access for training).

- Python environment (3.8 or later recommended)
- Access to a GPU is highly recommended for fine-tuning due to computational requirements.

## Dataset

### 1. Description of the dataset format and content.

A custom dataset of HTML product snippets and their corresponding JSON representations, loaded from 'json_extraction_dataset_500.json'. Each entry includes an 'input' HTML snippet and an 'output' JSON object. The data is expected to be in a JSON file named `json_extraction_dataset_500.json`.

### 2. How the data was preprocessed for fine-tuning.

The data is preprocessed by formatting each input-output pair into a conversational prompt structure. This involves combining the HTML `input` and the JSON `output` into a specific format that the language model is trained on, using system, user, and assistant roles.

## Model Fine-tuning

### 1. Base model used.

The base model used for fine-tuning is `unsloth/tinyllama-chat-bnb-4bit`.

### 2. Fine-tuning method (LoRA).

The model was fine-tuned using the LoRA (Low-Rank Adaptation) method.

### 3. Key hyperparameters used during training.

Key hyperparameters used during training include:
- Per Device Train Batch Size: 2
- Gradient Accumulation Steps: 4
- Warmup Steps: 10
- Number of Train Epochs: 3
- Learning Rate: 0.0002
- Optimizer: adamw_8bit
- Weight Decay: 0.01
- LR Scheduler Type: linear
- Seed: 3407

## Usage

### 1. How to load the fine-tuned model.

You can load the fine-tuned model and tokenizer using the `unsloth` and `transformers` libraries.
```python
from unsloth import FastLanguageModel
import torch
import json

model_name = "learn-abc/html-model-tinyllama-chat-bnb-4bit" # Your fine-tuned model repository ID
max_seq_length = 2048 # Or your chosen sequence length
dtype = None # Auto detection

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model) # Enable 2x faster inference
```

### 2. Example code for performing inference with the model.

Here's an example of how to use the loaded model for inference:
```python
messages = [
    {"role": "user", "content": "Extract the product information:\n<div class='product'><h2>iPad Air</h2><span class='price'>$1344</span><span class='category'>audio</span><span class='brand'>Dell</span></div>"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda") # Or "cpu" if not using GPU

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    use_cache=True,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)

response = tokenizer.batch_decode(outputs)[0]
print(response)
```

## Example Input-output
### Input
```html
Extract the product information:\n<div class='product'><h2>iPad Air</h2><span class='price'>$1344</span><span class='category'>audio</span><span class='brand'>Dell</span></div>
```
### Output
```json
{
  "name": "iPad Air",
  "price": "$1344",
  "category": "audio",
  "brand": "Dell"
}
```

## Contributing
* Fork the repository.
* Create a new branch (git checkout -b feature-branch).
* Make your changes.
* Commit your changes (git commit -m 'Add some feature').
* Push to the branch (git push origin feature-branch).
* Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Me
For any inquiries or support, please reach out to:

* **Name:** [Abhishek Singh](https://github.com/SinghIsWriting/)
* **LinkedIn:** [My LinkedIn Profile](https://www.linkedin.com/in/abhishek-singh-bba2662a9)
* **Portfolio:** [Abhishek Singh Portfolio](https://portfolio-abhishek-singh-nine.vercel.app/)

