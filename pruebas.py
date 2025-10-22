from models.deepseek_moe_16b.configuration_deepseek import DeepseekConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.deepseek_moe_16b.modeling_deepseek import DeepseekForCausalLM
model_name = "deepseek-ai/deepseek-moe-16b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
model = DeepseekForCausalLM.from_pretrained('simplified_models/deepseek-moe-16b-f3')
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
