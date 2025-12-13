from moeits.Qwen2MoE_simplification_service import Qwen2MoE_Simplification_Service
from moeits.models.qwen2_moe import Qwen2MoeForCausalLM
import torch


def test_layers_weights_qwen_simplification_service():
    factor = 5
    simp_service = Qwen2MoE_Simplification_Service("Qwen/Qwen1.5-MoE-A2.7B", factor=factor)
    simp_model = simp_service.simplify_original_model(mode='test', name = "qwen1.5")
    num_layers = len(simp_service.name_experts) == len(simp_model.model.layers)

    or_mod = simp_service.original_model
    expert_names = simp_service.name_experts

    same_weights = True

    for i in range(0, len(or_mod.model.layers)):
        names_experts = expert_names[i]
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].self_attn.q_proj.weight, or_mod.model.layers[i].self_attn.q_proj.weight)
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].self_attn.k_proj.weight, or_mod.model.layers[i].self_attn.k_proj.weight)
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].self_attn.v_proj.weight, or_mod.model.layers[i].self_attn.v_proj.weight)
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].self_attn.o_proj.weight, or_mod.model.layers[i].self_attn.o_proj.weight)

        same_weights = same_weights and torch.equal(simp_model.model.layers[i].mlp.gate.weight, or_mod.model.layers[i].mlp.gate.weight[names_experts,:])

        same_weights = same_weights and torch.equal(simp_model.model.layers[i].mlp.shared_expert.gate_proj.weight,or_mod.model.layers[i].mlp.shared_expert.gate_proj.weight)
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].mlp.shared_expert.up_proj.weight,or_mod.model.layers[i].mlp.shared_expert.up_proj.weight)
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].mlp.shared_expert.down_proj.weight,or_mod.model.layers[i].mlp.shared_expert.down_proj.weight)

        same_weights = same_weights and torch.equal(simp_model.model.layers[i].mlp.shared_expert_gate.weight,or_mod.model.layers[i].mlp.shared_expert_gate.weight)


        same_weights = same_weights and torch.equal(simp_model.model.layers[i].input_layernorm.weight,or_mod.model.layers[i].input_layernorm.weight)
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].post_attention_layernorm.weight,or_mod.model.layers[i].post_attention_layernorm.weight)

    same_weights = same_weights and torch.equal(simp_model.model.norm.weight,or_mod.model.norm.weight)
    same_weights = same_weights and torch.equal(simp_model.lm_head.weight,or_mod.lm_head.weight)

    expert_weights = True

    for i in range(0, len(or_mod.model.layers)):
            names_experts = expert_names[i]
            for j, e in enumerate(names_experts):
                expert_weights = expert_weights and torch.equal(simp_model.model.layers[i].mlp.experts[j].gate_proj.weight, or_mod.model.layers[i].mlp.experts[e].gate_proj.weight)
                expert_weights = expert_weights and torch.equal(simp_model.model.layers[i].mlp.experts[j].up_proj.weight, or_mod.model.layers[i].mlp.experts[e].up_proj.weight)
                expert_weights = expert_weights and torch.equal(simp_model.model.layers[i].mlp.experts[j].down_proj.weight, or_mod.model.layers[i].mlp.experts[e].down_proj.weight)


    assert num_layers and same_weights and expert_weights

def test_layers_weights_saving_loading_qwen_simplification_service():
    factor = 20
    simp_service = Qwen2MoE_Simplification_Service("Qwen/Qwen1.5-MoE-A2.7B", factor=factor)
    aux_model = simp_service.simplify_original_model(mode='prod', name = "qwen1.5")
    aux_model.save_pretrained(f'/MoEITS/simplified_models/qwen1.5-MoE-A2.7B-Chat-f{factor}-test/')

    simp_model = Qwen2MoeForCausalLM.from_pretrained(f'/MoEITS/simplified_models/qwen1.5-MoE-A2.7B-Chat-f{factor}-test/', device_map='auto')

    num_layers = len(simp_service.name_experts) == len(simp_model.model.layers)

    or_mod = simp_service.original_model
    expert_names = simp_service.name_experts

    same_weights = True

    for i in range(0, len(or_mod.model.layers)):
        names_experts = expert_names[i]
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].self_attn.q_proj.weight, or_mod.model.layers[i].self_attn.q_proj.weight)
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].self_attn.k_proj.weight, or_mod.model.layers[i].self_attn.k_proj.weight)
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].self_attn.v_proj.weight, or_mod.model.layers[i].self_attn.v_proj.weight)
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].self_attn.o_proj.weight, or_mod.model.layers[i].self_attn.o_proj.weight)

        same_weights = same_weights and torch.equal(simp_model.model.layers[i].mlp.gate.weight, or_mod.model.layers[i].mlp.gate.weight[names_experts,:])

        same_weights = same_weights and torch.equal(simp_model.model.layers[i].mlp.shared_expert.gate_proj.weight,or_mod.model.layers[i].mlp.shared_expert.gate_proj.weight)
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].mlp.shared_expert.up_proj.weight,or_mod.model.layers[i].mlp.shared_expert.up_proj.weight)
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].mlp.shared_expert.down_proj.weight,or_mod.model.layers[i].mlp.shared_expert.down_proj.weight)

        same_weights = same_weights and torch.equal(simp_model.model.layers[i].mlp.shared_expert_gate.weight,or_mod.model.layers[i].mlp.shared_expert_gate.weight)


        same_weights = same_weights and torch.equal(simp_model.model.layers[i].input_layernorm.weight,or_mod.model.layers[i].input_layernorm.weight)
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].post_attention_layernorm.weight,or_mod.model.layers[i].post_attention_layernorm.weight)

    same_weights = same_weights and torch.equal(simp_model.model.norm.weight,or_mod.model.norm.weight)
    same_weights = same_weights and torch.equal(simp_model.lm_head.weight,or_mod.lm_head.weight)

    expert_weights = True

    for i in range(0, len(or_mod.model.layers)):
            names_experts = expert_names[i]
            for j, e in enumerate(names_experts):
                expert_weights = expert_weights and torch.equal(simp_model.model.layers[i].mlp.experts[j].gate_proj.weight, or_mod.model.layers[i].mlp.experts[e].gate_proj.weight)
                expert_weights = expert_weights and torch.equal(simp_model.model.layers[i].mlp.experts[j].up_proj.weight, or_mod.model.layers[i].mlp.experts[e].up_proj.weight)
                expert_weights = expert_weights and torch.equal(simp_model.model.layers[i].mlp.experts[j].down_proj.weight, or_mod.model.layers[i].mlp.experts[e].down_proj.weight)


    assert num_layers and same_weights and expert_weights


#def test_layers_weights_deepseekv2_simplification_service():
     #pass
