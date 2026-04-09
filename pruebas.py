from models.deepseek_v2_lite.modeling_deepseek import DeepseekV2ForCausalLM
from models.deepseek_v2_lite.configuration_deepseek import DeepseekV2Config

from models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
from models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "deepseek-ai/DeepSeek-V2-Lite"
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
model = DeepseekV2ForCausalLM.from_pretrained('simplified_models/deepseek-v2-lite-f4.0',ignore_mismatched_sizes=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

from moeits.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
tok_path = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
model_path = "/MoEITS/simplified_models/qwen1.5-MoE-A2.7B-Chat-f15.0-mprod"
tokenizer = AutoTokenizer.from_pretrained(tok_path)
text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
model = Qwen2MoeForCausalLM.from_pretrained(model_path, device_map='auto')
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

# PARA TESTS

import torch

for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
    assert n1 == n2, (n1, n2)
    if not torch.allclose(p1, p2):
        print("Param distinto:", n1)

for (n1, b1), (n2, b2) in zip(model1.named_buffers(), model2.named_buffers()):
    assert n1 == n2, (n1, n2)
    if not torch.allclose(b1, b2):
        print("Buffer distinto:", n1)