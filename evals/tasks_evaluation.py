import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer
from moeits.models.qwen2_moe import Qwen2MoeForCausalLM
import torch
import json


model_path = "/MoEITS/training/models/prueba/qwen1.5-MoE-A2.7B-Chat-f1.25-mprod_retrained/"
tokenizer_path = "Qwen/Qwen1.5-MoE-A2.7B"


# 1. Load your custom model
model = Qwen2MoeForCausalLM.from_pretrained(model_path,device_map="auto",
                dtype=torch.float16, 
                trust_remote_code=True,
                attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)



# 2. Wrap it in the lm-eval HFLM class
# This acts as a bridge so lm-eval treats it like a standard HF model
lm_obj = HFLM(
    pretrained=model,
    tokenizer=tokenizer,
    batch_size=8
)

# 3. Run the evaluation programmatically
results = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=["hellaswag"],
    num_fewshot=5,
    batch_size=8,
    device="cuda:0"
)

clean_data = results["results"]

with open('output_evals/qwen1.5-MoE-A2.7B-Chat-f1.25-mprod_retrained_lmevals_hellaswag.json','w') as f:
    f.write(json.dumps(clean_data))