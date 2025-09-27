from MoEITS_simplification_service import MoEITS_Simplification_Service
from transformers import AutoTokenizer, MixtralForCausalLM, MixtralConfig, AutoModelForCausalLM
import json
import numpy as np
from utils import compute_information_measures




class Mixtral8x7b_Simplification_Service(MoEITS_Simplification_Service):
    def __init__(self, model_name, factor=1.5, output_base_path='', config_path='utils/config.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=config['token'])
        self.original_model = AutoModelForCausalLM.from_pretrained(self.model_name, token=config['token'], trust_remote_code=True, dtype="auto")
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
        