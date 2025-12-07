from moeits.Mixtral8x7b_simplification_service import Mixtral8x7b_Simplification_Service
import torch

def test_layers_weights_mixtral8x7b_simplification_service():
    factor = 5
    simp_service = Mixtral8x7b_Simplification_Service("mistralai/Mixtral-8x7B-Instruct-v0.1", factor=factor)
    simp_model = simp_service.simplify_original_model(mode='test', name = "mixtral")
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

        same_weights = same_weights and torch.equal(simp_model.model.layers[i].block_sparse_moe.gate.weight, or_mod.model.layers[i].block_sparse_moe.gate.weight[names_experts,:])
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].input_layernorm.weight,or_mod.model.layers[i].input_layernorm.weight)
        same_weights = same_weights and torch.equal(simp_model.model.layers[i].post_attention_layernorm.weight,or_mod.model.layers[i].post_attention_layernorm.weight)
 
    same_weights = same_weights and torch.equal(simp_model.model.norm.weight,or_mod.model.norm.weight)
    same_weights = same_weights and torch.equal(simp_model.lm_head.weight,or_mod.lm_head.weight)

    expert_weights = True

    for i in range(0, len(or_mod.model.layers)):
            names_experts = expert_names[i]
            for j, e in enumerate(names_experts):
                expert_weights = expert_weights and torch.equal(simp_model.model.layers[i].block_sparse_moe.experts[j].w1.weight, or_mod.model.layers[i].block_sparse_moe.experts[e].w1.weight)
                expert_weights = expert_weights and torch.equal(simp_model.model.layers[i].block_sparse_moe.experts[j].w2.weight, or_mod.model.layers[i].block_sparse_moe.experts[e].w2.weight)
                expert_weights = expert_weights and torch.equal(simp_model.model.layers[i].block_sparse_moe.experts[j].w3.weight, or_mod.model.layers[i].block_sparse_moe.experts[e].w3.weight)


    assert num_layers and same_weights and expert_weights
