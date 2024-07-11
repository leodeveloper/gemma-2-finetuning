from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
device = "mps" if torch.has_mps else "cpu"
print(device)

os.environ['HF_TOKEN']=""

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b",
    device_map=device,
    torch_dtype=torch.bfloat16
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))