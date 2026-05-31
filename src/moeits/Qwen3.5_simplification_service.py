from moeits.MoEITS_simplification_service import MoEITS_Simplification_Service
from moeits.utils import compute_information_measures_torch
import torch
from huggingface_hub import hf_hub_download  
import json



class Qwen3_5_Simplification_Service(MoEITS_Simplification_Service):
    def __init__(self, model_name, factor = None, output_base_path='', auth_path='/MoEITS/utils/config.json', nmi_base_path = '/MoEITS/NMI_matrices/', number_of_experts = None):
        with open(auth_path, 'r') as f:
            auth = json.load(f)
        self.nmi_base_path = nmi_base_path    
        self.model_name = model_name
        self.config_model = hf_hub_download(repo_id=self.model_name, filename="config.json")
        self.safetensor_index = hf_hub_download(repo_id=self.model_name, filename="model.safetensors.index.json")
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
            num_layers = len(self.original_model.model.layers)
            for i in range(num_layers):
                experts = self.original_model.model.layers[i].mlp.experts
                nmi = self._calculate_NMI_experts(experts)
                self.layers['L_'+str(i)] = nmi
            self._save_NMI_matrix(name)
    
    #TODO
    def _calculate_NMI_experts(self, experts):
        results = torch.zeros((len(experts), len(experts)))
        for i, e1 in enumerate(experts):
            for j, e2 in enumerate(experts):
                if i<j:
                    gate_info = compute_information_measures_torch(experts[i].gate_proj.weight.detach().cpu().float().numpy(), experts[j].gate_proj.weight.detach().cpu().float().numpy())['NMI']
                    up_info = compute_information_measures_torch(experts[i].up_proj.weight.detach().cpu().float().numpy(), experts[j].up_proj.weight.detach().cpu().float().numpy())['NMI']
                    down_info = compute_information_measures_torch(experts[i].down_proj.weight.detach().cpu().float().numpy(), experts[j].down_proj.weight.detach().cpu().float().numpy())['NMI']
                    # Weighting more up and down info in terms of redundancy
                    # Revisar para QWen
                    results[i,j] = (gate_info*0.2+up_info*0.4+down_info*0.4)
                    results[j,i] = (gate_info*0.2+up_info*0.4+down_info*0.4)
        
        return results
