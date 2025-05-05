import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载训练后的策略模型和Tokenzier
loaded_model = AutoModelForCausalLM.from_pretrained("./ppo_trained_model").to(device)
loaded_tokenizer = AutoTokenizer.from_pretrained("./ppo_trained_model")

# 生成示例
inputs = loaded_tokenizer("请问1+1等于多少？", return_tensors="pt").to(device)
outputs = loaded_model.generate(**inputs, max_new_tokens=200)
print(loaded_tokenizer.decode(outputs[0], skip_special_tokens=True))