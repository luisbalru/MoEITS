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
num_experts = list(np.arange(64,64-26,-1))
# DEFINICIÓN EXACTO DE deepseek-ai/deepseek-moe-16b-base
model = DeepseekForCausalLM(DeepseekConfig(n_routed_experts=64,
                                           num_experts_by_block=num_experts,
                                           hidden_size=2048,
                                           n_shared_experts=2, 
                                           num_hidden_layers=28, 
                                           num_experts_per_tok=6, 
                                           first_k_dense_replace=1,
                                           attention_bias=False,
                                           attention_dropout=0.0,
                                           bos_token_id=100000,
                                           eos_token_id=100001,
                                           hidden_act='silu',
                                           initializer_range=0.02,
                                           intermediate_size=10944,
                                           max_position_embeddings=4096,
                                           moe_intermediate_size=1408,
                                           moe_layer_freq=1,
                                           norm_topk_prob=False,
                                           num_attention_heads=16,
                                           num_key_value_heads=16,
                                           pretraining_tp=1,
                                           rms_norm_eps=1e-06,
                                           rope_scaling=None,
                                           rope_theta=10000,
                                           scoring_func='softmax',
                                           tie_word_embeddings=False,
                                           use_cache=True,
                                           vocab_size=102400))


print(model)
print(f"Total parameters: {model.num_parameters()}")

#model2 = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-moe-16b-base", trust_remote_code=True, torch_dtype="auto")
#print(f"Total parameters from_pretrained: {model2.num_parameters()}")


"""
#moe_simp_service = Mixtral8x7b_Simplification_Service("mistralai/Mixtral-8x7B-Instruct-v0.1")
moe_simp_service = DeepSeekMoE_Simplification_Service("deepseek-ai/deepseek-moe-16b-base")
print("Parámetros modelo original: ", np.sum(p.numel() for p in moe_simp_service.original_model.parameters() if p.requires_grad))
simp = moe_simp_service.simplify_original_model()
print("Parámetros modelo simplificado: ", np.sum(p.numel() for p in simp.parameters() if p.requires_grad))
input()
print(simp)
"""