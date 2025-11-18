from moeits.DeepSeekV2Lite_simplification_service import DeepSeekV2Lite_Simplification_Service
from moeits.Mixtral8x7b_simplification_service import Mixtral8x7b_Simplification_Service
from moeits.Qwen2MoE_simplification_service import Qwen2MoE_Simplification_Service



def test_qwen_simplification_service():
    factor = 5
    simp_service = Qwen2MoE_Simplification_Service("Qwen/Qwen1.5-MoE-A2.7B", factor=factor)
    simp_model = simp_service.simplify_original_model(mode='test', name = "qwen")
    num_layers = len(simp_service.name_experts) == len(simp_model.model.layers)

    or_mod = simp_service.original_model
    expert_names = simp_service.name_experts

    same_weights = True

    for i in range(0, len(or_mod.model.layers)):
        names_experts = expert_names[i]
        same_weights = same_weights and simp_model.model.layers[i].self_attn.q_proj.weight == or_mod.model.layers[i].self_attn.q_proj.weight
        same_weights = same_weights and simp_model.model.layers[i].self_attn.k_proj.weight == or_mod.model.layers[i].self_attn.k_proj.weight
        same_weights = same_weights and simp_model.model.layers[i].self_attn.v_proj.weight == or_mod.model.layers[i].self_attn.v_proj.weight
        same_weights = same_weights and simp_model.model.layers[i].self_attn.o_proj.weight == or_mod.model.layers[i].self_attn.o_proj.weight

        same_weights = same_weights and simp_model.model.layers[i].mlp.gate.weight == or_mod.model.layers[i].mlp.gate.weight[names_experts,:]

        same_weights = same_weights and simp_model.model.layers[i].mlp.shared_expert.gate_proj.weight == or_mod.model.layers[i].mlp.shared_expert.gate_proj.weight
        same_weights = same_weights and simp_model.model.layers[i].mlp.shared_expert.up_proj.weight == or_mod.model.layers[i].mlp.shared_expert.up_proj.weight
        same_weights = same_weights and simp_model.model.layers[i].mlp.shared_expert.down_proj.weight == or_mod.model.layers[i].mlp.shared_expert.down_proj.weight

        same_weights = same_weights and simp_model.model.layers[i].mlp.shared_expert_gate.weight == or_mod.model.layers[i].mlp.shared_expert_gate.weight


        same_weights = same_weights and simp_model.model.layers[i].input_layernorm.weight == or_mod.model.layers[i].input_layernorm.weight
        same_weights = same_weights and simp_model.model.layers[i].post_attention_layernorm.weight == or_mod.model.layers[i].post_attention_layernorm.weight
    
    same_weights = same_weights and simp_model.model.norm.weight == or_mod.model.norm.weight
    same_weights = same_weights and simp_model.lm_head.weight == or_mod.lm_head.weight




    assert num_layers and same_weights