from moeits.MoEITS_simplification_service import MoEITS_Simplification_Service
from moeits.utils import compute_pairwise_nmi_matrix
from safetensors import safe_open   
import torch
from huggingface_hub import hf_hub_download  
import json
import os
import numpy as np



class Qwen3_5_Simplification_Service(MoEITS_Simplification_Service):
    def __init__(self, model_name, factor = None, output_base_path='', auth_path='/MoEITS/utils/config.json', nmi_base_path = '/MoEITS/NMI_matrices/', number_of_experts = None):
        with open(auth_path, 'r') as f:
            auth = json.load(f)
        self.nmi_base_path = nmi_base_path    
        self.model_name = model_name
        self.config_model = hf_hub_download(repo_id=self.model_name, filename="config.json")
        self.safetensor_index = hf_hub_download(repo_id=self.model_name, filename="model.safetensors.index.json")
        with open(self.safetensor_index, "r") as f:               
            self.weight_map = json.load(f)["weight_map"]  
        self.output_base_path = output_base_path
        self.number_of_experts = number_of_experts
        

    # TODO
    def _get_mutual_information_metrics(self, name):
        nmi_info = os.listdir(self.nmi_base_path)
        if name+'.npz' in nmi_info:
            print("Loading NMI metrics...")
            self.layers = dict(np.load(os.path.join(self.nmi_base_path, name+'.npz')))
        else:
            print("Calculating NMI metrics...")
            num_layers = self.config_model["num_hidden_layers"]
            for i in range(num_layers):
                nmi = self._calculate_NMI_experts(i)
                self.layers['L_'+str(i)] = nmi.numpy()
            self._save_NMI_matrix(name)
    
    def _calculate_NMI_experts(self, idx):
        tensor_names = [f"model.language_model.layers.{idx}.mlp.experts.gate_up_proj", f"model.language_model.layers.{idx}.mlp.experts.down_proj"]
        nmis = []
        for t in tensor_names:
            shard_filename = self.weight_map[t] 
            shard_path = hf_hub_download(repo_id=self.model_name, filename=shard_filename) 
            with safe_open(shard_path, framework="pt", device="cuda") as shard_file: 
                weights = shard_file.get_tensor(t) 
                nmis.append(compute_pairwise_nmi_matrix(weights))

        return 0.5*nmis[0] + 0.5*nmis[1]
    
    def _build_simplified_model(self):
        pass

    def _set_weights_to_experts(self):
        pass

    def _set_weights_to_new_model(self):
        pass