#from MoEITS_simplification_service import MoEITS_Simplification_Service
from Mixtral8x7b_simplification_service import Mixtral8x7b_Simplification_Service
from DeepSeekMoE_simplification_service import DeepSeekMoE_Simplification_Service
import numpy as np
from transformers import AutoTokenizer, MixtralForCausalLM, MixtralConfig, DeepseekV3ForCausalLM,DeepseekV3Config

"""
#moe_simp_service = Mixtral8x7b_Simplification_Service("mistralai/Mixtral-8x7B-Instruct-v0.1")
moe_simp_service = DeepSeekMoE_Simplification_Service("deepseek-ai/deepseek-moe-16b-base")
print("Parámetros modelo original: ", np.sum(p.numel() for p in moe_simp_service.original_model.parameters() if p.requires_grad))
simp = moe_simp_service.simplify_original_model()
print("Parámetros modelo simplificado: ", np.sum(p.numel() for p in simp.parameters() if p.requires_grad))
input()
print(simp)
"""
ds = DeepseekV3ForCausalLM(DeepseekV3Config())
print(ds)