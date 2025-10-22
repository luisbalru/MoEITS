from MoEITS_simplification_service import MoEITS_Simplification_Service
from models.deepseek_moe_16b.modeling_deepseek import DeepseekForCausalLM
from models.deepseek_moe_16b.configuration_deepseek import DeepseekConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from utils import compute_information_measures
import torch




class DeepSeekMoE_Simplification_Service(MoEITS_Simplification_Service):
    def __init__(self, model_name, factor=1.5, output_base_path='', config_path='utils/config.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=config['token'])
        self.original_model = AutoModelForCausalLM.from_pretrained(self.model_name, token=config['token'], trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
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
                    results[i,j] = (gate_info+up_info+down_info)/3
                    results[j,i] = (gate_info+up_info+down_info)/3
        
        return results

    def _build_simplified_model(self, num_experts, name_experts):
        self.simplified_model = DeepseekForCausalLM(DeepseekConfig(n_routed_experts=64,
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

    def _set_weights_to_new_model(self, names):
        print("Embedding tokens")
        self.simplified_model.model.embed_tokens.weight = self.original_model.model.embed_tokens.weight
        print("Layers")

        self.simplified_model.model.layers[0].self_attn.q_proj.weight = self.original_model.model.layers[0].self_attn.q_proj.weight
        self.simplified_model.model.layers[0].self_attn.k_proj.weight = self.original_model.model.layers[0].self_attn.k_proj.weight
        self.simplified_model.model.layers[0].self_attn.v_proj.weight = self.original_model.model.layers[0].self_attn.v_proj.weight
        self.simplified_model.model.layers[0].self_attn.o_proj.weight = self.original_model.model.layers[0].self_attn.o_proj.weight
        #self.simplified_model.model.layers[0].self_attn.rotary_emb.weight = self.original_model.model.layers[0].self_attn.rotary_emb.weight

        self.simplified_model.model.layers[0].mlp.gate_proj.weight = self.original_model.model.layers[0].mlp.gate_proj.weight
        self.simplified_model.model.layers[0].mlp.up_proj.weight = self.original_model.model.layers[0].mlp.up_proj.weight
        self.simplified_model.model.layers[0].mlp.down_proj.weight = self.original_model.model.layers[0].mlp.down_proj.weight

        self.simplified_model.model.layers[0].input_layernorm.weight = self.original_model.model.layers[0].input_layernorm.weight
        self.simplified_model.model.layers[0].post_attention_layernorm.weight = self.original_model.model.layers[0].post_attention_layernorm.weight


        for i in range(1, len(self.original_model.model.layers)):
            names_experts = names[i-1]
            self.simplified_model.model.layers[i].self_attn.q_proj.weight = self.original_model.model.layers[i].self_attn.q_proj.weight
            self.simplified_model.model.layers[i].self_attn.k_proj.weight = self.original_model.model.layers[i].self_attn.k_proj.weight
            self.simplified_model.model.layers[i].self_attn.v_proj.weight = self.original_model.model.layers[i].self_attn.v_proj.weight
            self.simplified_model.model.layers[i].self_attn.o_proj.weight = self.original_model.model.layers[i].self_attn.o_proj.weight
            #self.simplified_model.model.layers[i].self_attn.rotary_emb.weight = self.original_model.model.layers[i].self_attn.rotary_emb.weight

            self.simplified_model.model.layers[i].mlp.gate.weight = torch.nn.Parameter(self.original_model.model.layers[i].mlp.gate.weight[names_experts,:])

            self.simplified_model.model.layers[i].mlp.shared_experts.gate_proj.weight = self.original_model.model.layers[i].mlp.shared_experts.gate_proj.weight
            self.simplified_model.model.layers[i].mlp.shared_experts.up_proj.weight = self.original_model.model.layers[i].mlp.shared_experts.up_proj.weight
            self.simplified_model.model.layers[i].mlp.shared_experts.down_proj.weight = self.original_model.model.layers[i].mlp.shared_experts.down_proj.weight

            self.simplified_model.model.layers[i].input_layernorm.weight = self.original_model.model.layers[i].input_layernorm.weight
            self.simplified_model.model.layers[i].post_attention_layernorm.weight = self.original_model.model.layers[i].post_attention_layernorm.weight
        
        self.simplified_model.model.norm.weight = self.original_model.model.norm.weight
        self.simplified_model.lm_head.weight = self.original_model.lm_head.weight

 
    def _set_weights_to_experts(self, names):
        # First layer doesn't have experts and the 27 following ones do have
        for i in range(1, len(self.original_model.layers)):
            names_experts = names[i-1]
            for j, e in enumerate(names_experts):
                self.simplified_model.model.layers[i].mlp.experts[j].gate_proj.weight = self.original_model.model.layers[i].mlp.experts[e].gate_proj.weight
                self.simplified_model.model.layers[i].mlp.experts[j].up_proj.weight = self.original_model.model.layers[i].mlp.experts[e].up_proj.weight
                self.simplified_model.model.layers[i].mlp.experts[j].down_proj.weight = self.original_model.model.layers[i].mlp.experts[e].down_proj.weight