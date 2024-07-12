from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime

print(datetime.now().time())

HF_TOKEN = os.environ.get("HF_TOKEN")

device = "mps" if torch.backends.mps.is_built() else "cpu"

os.environ['HF_TOKEN']=HF_TOKEN

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2",resume_download=None)
model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2",
    device_map=device,
    torch_dtype=torch.bfloat16,
    resume_download=None
)

model.generation_config.pad_token_id = model.generation_config.eos_token_id

input_text = "Tell me some thing about Karachi Pakistan."
input_ids = tokenizer(input_text, return_tensors="pt", truncation=True ).to(device)
#UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.85` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
outputs = model.generate(**input_ids,
                         do_sample=True,
                          #max_length=200,
                          max_new_tokens=500,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.8,
        top_p=0.85
                         #,  max_new_tokens=1000
                         )#, pad_token_id=tokenizer.eos_token_id)
#print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
#print("------------------------------------")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
#print(tokenizer.decode(outputs[0]))
print(datetime.now().time())
