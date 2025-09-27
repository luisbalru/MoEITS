from MoEITS_simplification_service import MoEITS_Simplification_Service
from transformers import AutoTokenizer, MixtralForCausalLM, MixtralConfig, AutoModelForCausalLM
import json
import numpy as np
from utils import compute_information_measures
import torch




class Mixtral8x7b_Simplification_Service(MoEITS_Simplification_Service):
    def __init__(self, model_name, factor=1.5, output_base_path='', config_path='utils/config.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=config['token'])
        #self.original_model = AutoModelForCausalLM.from_pretrained(self.model_name, token=config['token'], trust_remote_code=True, dtype="auto")
        self.original_model = AutoModelForCausalLM.from_pretrained(self.model_name, token=config['token'])
        print(self.original_model)
        self.layers = {}
        self.factor = factor
        self.output_base_path = output_base_path
        self.original_model = None

    def _get_mutual_information_metrics(self):
        print("Getting NMI metrics...")
        num_layers = len(self.original_model.model.layers)
        for i in range(num_layers):
            experts = self.original_model.model.layers[i].block_sparse_moe.experts
            nmi = self._calculate_NMI_experts(experts)
            self.layers['L_'+str(i)] = nmi
    
    def _calculate_NMI_experts(self, experts):
        results = np.zeros((len(experts), len(experts)))
        for i, e1 in enumerate(experts):
            for j, e2 in enumerate(experts):
                if i<j:
                    w1_info = compute_information_measures(experts[e1].w1.weight.detach().numpy(), experts[e2].w1.weight.detach().numpy())['NMI']
                    w2_info = compute_information_measures(experts[e1].w2.weight.detach().numpy(), experts[e2].w2.weight.detach().numpy())['NMI']
                    w3_info = compute_information_measures(experts[e1].w3.weight.detach().numpy(), experts[e2].w3.weight.detach().numpy())['NMI']
                    results[i,j] = (w1_info+w2_info+w3_info)/3
                    results[j,i] = (w1_info+w2_info+w3_info)/3
        
        return results

    def _build_simplified_model(self, num_experts, name_experts):
        self.simplified_model = MixtralForCausalLM(MixtralConfig(max_position_embeddings=32768, name_experts_by_block=name_experts, num_experts_by_block=num_experts))

    def _set_weights_to_new_model(self, names):
        print("Embedding tokens")
        self.simplified_model.model.embed_tokens.weight = self.original_model.model.embed_tokens.weight
        print("Layers")
        for i in range(len(self.original_model.model.layers)):
            names_experts = names[i]
            self.simplified_model.model.layers[i].self_attn.q_proj.weight = self.original_model.model.layers[i].self_attn.q_proj.weight
            self.simplified_model.model.layers[i].self_attn.k_proj.weight = self.original_model.model.layers[i].self_attn.k_proj.weight
            self.simplified_model.model.layers[i].self_attn.v_proj.weight = self.original_model.model.layers[i].self_attn.v_proj.weight
            self.simplified_model.model.layers[i].self_attn.o_proj.weight = self.original_model.model.layers[i].self_attn.o_proj.weight

            self.simplified_model.model.layers[i].block_sparse_moe.gate.weight = torch.nn.Parameter(self.original_model.model.layers[i].block_sparse_moe.gate.weight[names_experts,:])

            self.simplified_model.model.layers[i].input_layernorm.weight = self.original_model.model.layers[i].input_layernorm.weight
            self.simplified_model.model.layers[i].post_attention_layernorm.weight = self.original_model.model.layers[i].post_attention_layernorm.weight
        self.simplified_model.model.norm.weight = self.original_model.model.norm.weight
        self.simplified_model.lm_head.weight = self.original_model.lm_head.weight
 
    def _set_weights_to_experts(self, names):
        for i in range(len(names)):
            names_experts = names[i]
            for j, e in enumerate(names_experts):
                self.simplified_model.model.layers[i].block_sparse_moe.experts[j].w1.weight = self.original_model.model.layers[i].block_sparse_moe.experts[e].w1.weight
                self.simplified_model.model.layers[i].block_sparse_moe.experts[j].w2.weight = self.original_model.model.layers[i].block_sparse_moe.experts[e].w2.weight
                self.simplified_model.model.layers[i].block_sparse_moe.experts[j].w3.weight = self.original_model.model.layers[i].block_sparse_moe.experts[e].w3.weight