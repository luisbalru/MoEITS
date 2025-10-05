#from MoEITS_simplification_service import MoEITS_Simplification_Service
from Mixtral8x7b_simplification_service import Mixtral8x7b_Simplification_Service
from DeepSeekMoE_simplification_service import DeepSeekMoE_Simplification_Service
import numpy as np
from transformers import AutoTokenizer, MixtralForCausalLM, MixtralConfig, DeepseekV3ForCausalLM,DeepseekV3Config
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

# 1. Define the configuration for deepseek-moe-16b
# These parameters are taken directly from the model's official config.json file.
config = DeepseekV2Config(
    # Core Architecture
    hidden_size=5120,
    intermediate_size=14336,
    num_hidden_layers=28,
    num_attention_heads=128,
    num_key_value_heads=128, # Standard Multi-Head Attention (not GQA)
    hidden_act="silu",
    max_position_embeddings=16384,
    vocab_size=102400,

    # MoE (Mixture of Experts) Specific Parameters 🧠
    n_routed_experts=64,
    num_experts_per_tok=2, # The "k" in Top-K routing
    moe_intermediate_size=2560,
    n_shared_experts=2,
    routed_scaling_factor=1.0,

    # Other important parameters
    rms_norm_eps=1e-6,
    rope_theta=100000.0,
    initializer_range=0.02,
    attention_bias=False,
    attention_dropout=0.0,
    
    # Model Identifier
    model_type="deepseek_v2",
    torch_dtype="bfloat16", # Specify the desired data type
)

# 2. Instantiate the model from the configuration object 🏗️
model = DeepSeekV2ForCausalLM(config)

# You now have the model architecture built in memory.
print(model)
print(f"Total parameters: {model.num_parameters() / 1e9:.2f}B")


"""
#moe_simp_service = Mixtral8x7b_Simplification_Service("mistralai/Mixtral-8x7B-Instruct-v0.1")
moe_simp_service = DeepSeekMoE_Simplification_Service("deepseek-ai/deepseek-moe-16b-base")
print("Parámetros modelo original: ", np.sum(p.numel() for p in moe_simp_service.original_model.parameters() if p.requires_grad))
simp = moe_simp_service.simplify_original_model()
print("Parámetros modelo simplificado: ", np.sum(p.numel() for p in simp.parameters() if p.requires_grad))
input()
print(simp)
"""