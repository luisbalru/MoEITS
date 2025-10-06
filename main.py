#from MoEITS_simplification_service import MoEITS_Simplification_Service
#from Mixtral8x7b_simplification_service import Mixtral8x7b_Simplification_Service
#from DeepSeekMoE_simplification_service import DeepSeekMoE_Simplification_Service
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, MixtralForCausalLM, MixtralConfig, DeepseekV3ForCausalLM,DeepseekV3Config
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM
from models.deepseek_moe_16b.modeling_deepseek import DeepseekForCausalLM
from models.deepseek_moe_16b.configuration_deepseek import DeepseekConfig

"""
# 1. Define the configuration for deepseek-moe-16b
# These parameters are taken directly from the model's official config.json file.
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-moe-16b-base", trust_remote_code=True, torch_dtype="auto")
config = model.config

# 2. Instantiate the model from the configuration object 🏗️
model = DeepseekV3ForCausalLM(config)

# You now have the model architecture built in memory.
print(model)
print(f"Total parameters: {model.num_parameters() / 1e9:.2f}B")
"""

model = DeepseekForCausalLM(DeepseekConfig(n_routed_experts=64, hidden_size=2048,shared_experts=2, num_attention_head=16, num_experts_per_tok=6, first_k_dense_replace=1))
print(model)


"""
#moe_simp_service = Mixtral8x7b_Simplification_Service("mistralai/Mixtral-8x7B-Instruct-v0.1")
moe_simp_service = DeepSeekMoE_Simplification_Service("deepseek-ai/deepseek-moe-16b-base")
print("Parámetros modelo original: ", np.sum(p.numel() for p in moe_simp_service.original_model.parameters() if p.requires_grad))
simp = moe_simp_service.simplify_original_model()
print("Parámetros modelo simplificado: ", np.sum(p.numel() for p in simp.parameters() if p.requires_grad))
input()
print(simp)
"""