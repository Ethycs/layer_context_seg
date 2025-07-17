---
license: creativeml-openrail-m
datasets:
- amphora/QwQ-LongCoT-130K
language:
- en
base_model:
- Qwen/Qwen2.5-7B-Instruct
pipeline_tag: text-generation
library_name: transformers
tags:
- Long-CoT
- Qwen2.5
- 7B
- safetensors
- text-generation-inference
- QwQ
- SFT
- Math
- Qwen with Questions
new_version: prithivMLmods/QwQ-LCoT2-7B-Instruct
---

# **QwQ-LCoT-7B-Instruct Model File**

The QwQ-LCoT-7B-Instruct is a fine-tuned language model designed for advanced reasoning and instruction-following tasks. It leverages the Qwen2.5-7B base model and has been fine-tuned on the amphora/QwQ-LongCoT-130K dataset, focusing on chain-of-thought (CoT) reasoning. This model is optimized for tasks requiring logical reasoning, detailed explanations, and multi-step problem-solving, making it ideal for applications such as instruction-following, text generation, and complex reasoning tasks.

## Quickstart with Transformers

Here provides a code snippet with `apply_chat_template` to show you how to load the tokenizer and model and how to generate contents.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "prithivMLmods/QwQ-LCoT-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "How many r in strawberry."
messages = [
    {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### **Sample Long CoT:**

![Screenshot 2024-12-13 211732.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Mgm9LmQZlFZmglKYwEDYA.png)

---
### **Key Features:**

1. **Model Size:**  
   - **7.62B parameters** (FP16 precision).  

2. **Model Sharding:**  
   - The model weights are split into 4 shards (`safetensors`) for efficient storage and download:
     - `model-00001-of-00004.safetensors` (4.88 GB)
     - `model-00002-of-00004.safetensors` (4.93 GB)
     - `model-00003-of-00004.safetensors` (4.33 GB)
     - `model-00004-of-00004.safetensors` (1.09 GB)

3. **Tokenizer:**  
   - Byte-pair encoding (BPE) based.
   - Files included:
     - `vocab.json` (2.78 MB)
     - `merges.txt` (1.82 MB)
     - `tokenizer.json` (11.4 MB)
   - Special tokens mapped in `special_tokens_map.json` (e.g., `<pad>`, `<eos>`).

4. **Configuration Files:**  
   - `config.json`: Defines model architecture and hyperparameters.
   - `generation_config.json`: Settings for inference and text generation tasks.

---

### **Training Dataset:**  
- **Dataset Name:** [amphora/QwQ-LongCoT-130K](https://huggingface.co/datasets/amphora/QwQ-LongCoT-130K)  
- **Size:** 133k examples.  
- **Focus:** Chain-of-Thought reasoning for complex tasks.

---

### **Use Cases:**
1. **Instruction Following:**  
   Handle user instructions effectively, even for multi-step tasks.
   
2. **Reasoning Tasks:**  
   Perform logical reasoning and generate detailed step-by-step solutions.
   
3. **Text Generation:**  
   Generate coherent, context-aware responses.
---