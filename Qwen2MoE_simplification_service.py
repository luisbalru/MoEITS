from MoEITS_simplification_service import MoEITS_Simplification_Service
from models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
from models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from utils import compute_information_measures
import torch




class Qwen2MoE_Simplification_Service(MoEITS_Simplification_Service):
    def __init__(self, model_name, factor=1.5, output_base_path='', auth_path='utils/config.json'):
        with open(auth_path, 'r') as f:
            auth = json.load(f)
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.original_model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        self.config_model = self.original_model.config.to_dict()
        self.layers = {}
        self.factor = factor
        self.output_base_path = output_base_path

    def _get_mutual_information_metrics(self):
        print("Getting NMI metrics...")
        num_layers = len(self.original_model.model.layers)
        for i in range(1, num_layers):
            experts = self.original_model.model.layers[i].mlp.experts
            nmi = self._calculate_NMI_experts(experts)
            self.layers['L_'+str(i)] = nmi
    
    def _calculate_NMI_experts(self, experts):
        results = np.zeros((len(experts), len(experts)))
        for i, e1 in enumerate(experts):
            for j, e2 in enumerate(experts):
                if i<j:
                    gate_info = compute_information_measures(experts[i].gate_proj.weight.detach().cpu().float().numpy(), experts[j].gate_proj.weight.detach().cpu().float().numpy())['NMI']
                    up_info = compute_information_measures(experts[i].up_proj.weight.detach().cpu().float().numpy(), experts[j].up_proj.weight.detach().cpu().float().numpy())['NMI']
                    down_info = compute_information_measures(experts[i].down_proj.weight.detach().cpu().float().numpy(), experts[j].down_proj.weight.detach().cpu().float().numpy())['NMI']
                    # Weighting more up and down info in terms of redundancy
                    # Revisar para QWen
                    results[i,j] = (gate_info*0.2+up_info*0.4+down_info*0.4)
                    results[j,i] = (gate_info*0.2+up_info*0.4+down_info*0.4)
        
        return results

    def _build_simplified_model(self, num_experts, name_experts):
        self.config_model['num_experts_by_block'] = num_experts
        self.simplified_model = Qwen2MoeForCausalLM(Qwen2MoeConfig(**self.config_model))

    def _set_weights_to_new_model(self, names):
        print("Embedding tokens")
        self.simplified_model.model.embed_tokens.weight = self.original_model.model.embed_tokens.weight
        print("Layers")

        for i in range(1, len(self.original_model.model.layers)):
            names_experts = names[i-1]
            self.simplified_model.model.layers[0].self_attn.q_proj.weight = self.original_model.model.layers[0].self_attn.q_proj.weight
            self.simplified_model.model.layers[0].self_attn.k_proj.weight = self.original_model.model.layers[0].self_attn.k_proj.weight
            self.simplified_model.model.layers[0].self_attn.v_proj.weight = self.original_model.model.layers[0].self_attn.v_proj.weight
            self.simplified_model.model.layers[0].self_attn.o_proj.weight = self.original_model.model.layers[0].self_attn.o_proj.weight

            self.simplified_model.model.layers[i].mlp.gate.weight = torch.nn.Parameter(self.original_model.model.layers[i].mlp.gate.weight[names_experts,:])

            self.simplified_model.model.layers[i].mlp.shared_expert.gate_proj.weight = self.original_model.model.layers[i].mlp.shared_expert.gate_proj.weight
            self.simplified_model.model.layers[i].mlp.shared_expert.up_proj.weight = self.original_model.model.layers[i].mlp.shared_expert.up_proj.weight
            self.simplified_model.model.layers[i].mlp.shared_expert.down_proj.weight = self.original_model.model.layers[i].mlp.shared_expert.down_proj.weight

            self.simplified_model.model.layers[i].mlp.shared_expert_gate.weight = self.original_model.model.layers[i].mlp.shared_expert_gate.weight


            self.simplified_model.model.layers[i].input_layernorm.weight = self.original_model.model.layers[i].input_layernorm.weight
            self.simplified_model.model.layers[i].post_attention_layernorm.weight = self.original_model.model.layers[i].post_attention_layernorm.weight
        
        self.simplified_model.model.norm.weight = self.original_model.model.norm.weight
        self.simplified_model.lm_head.weight = self.original_model.lm_head.weight

 
    def _set_weights_to_experts(self, names):
        for i in range(0, len(self.original_model.model.layers)):
            names_experts = names[i-1]
            for j, e in enumerate(names_experts):
                self.simplified_model.model.layers[i].mlp.experts[j].gate_proj.weight = self.original_model.model.layers[i].mlp.experts[e].gate_proj.weight
                self.simplified_model.model.layers[i].mlp.experts[j].up_proj.weight = self.original_model.model.layers[i].mlp.experts[e].up_proj.weight
                self.simplified_model.model.layers[i].mlp.experts[j].down_proj.weight = self.original_model.model.layers[i].mlp.experts[e].down_proj.weight