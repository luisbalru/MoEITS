import numpy as np
from scipy.stats import entropy, iqr
import pandas as pd
import os
from numpy import unravel_index
import json
from abc import ABC, abstractmethod
from copy import deepcopy


class MoEITS_Simplification_Service(ABC):
    @abstractmethod
    def _get_mutual_information_metrics(self, name):
        pass
    
    @abstractmethod
    def _build_simplified_model(self, num_experts, name_experts):
        pass

    @abstractmethod
    def _calculate_NMI_experts(self, experts):
        pass

    @abstractmethod
    def _set_weights_to_new_model(self, names):
        pass

    @abstractmethod
    def _set_weights_to_experts(self, names):
        pass

    def _save_NMI_matrix(self, name):
        print(f"Saving NMI info to file {os.path.join(self.nmi_base_path, name+'.npz')}...")
        np.savez_compressed(os.path.join(self.nmi_base_path, name+'.npz'), **self.layers)


    def _simplify_block(self, nmi_matrix):
        remaining_experts = np.arange(nmi_matrix.shape[0]).tolist()
        mean = np.mean(nmi_matrix)
        iqr_val = iqr(nmi_matrix)
        th = mean + iqr_val*self.factor
        while len(np.where(nmi_matrix > th)[0]):
            arg_e1, arg_e2 = unravel_index(np.argmax(nmi_matrix), nmi_matrix.shape)
            closeness_e1 = np.mean(nmi_matrix[arg_e1,:])
            closeness_e2 = np.mean(nmi_matrix[arg_e2,:])
            arg_expert_remove = arg_e1 if closeness_e1 > closeness_e2 else arg_e2
            remaining_experts.pop(arg_expert_remove)
            nmi_matrix = np.delete(np.delete(nmi_matrix, arg_expert_remove, axis=0), arg_expert_remove, axis=1)
        return remaining_experts
    
    def _simplify_block_fixed_number_of_experts(self, nmi_matrix):
        remaining_experts = np.arange(nmi_matrix.shape[0]).tolist()
        while len(remaining_experts) > self.number_of_experts:
            arg_e1, arg_e2 = unravel_index(np.argmax(nmi_matrix), nmi_matrix.shape)
            closeness_e1 = np.mean(nmi_matrix[arg_e1,:])
            closeness_e2 = np.mean(nmi_matrix[arg_e2,:])
            arg_expert_remove = arg_e1 if closeness_e1 > closeness_e2 else arg_e2
            remaining_experts.pop(arg_expert_remove)
            nmi_matrix = np.delete(np.delete(nmi_matrix, arg_expert_remove, axis=0), arg_expert_remove, axis=1)
        return remaining_experts

    def _simplify_model(self):
        experts = []
        name_experts = []

        print("Simplifying model...")
        for k in self.layers.keys():
            nmi_encoder = self.layers[k]
            if self.number_of_experts:
                remaining_experts = self._simplify_block_fixed_number_of_experts(nmi_encoder)
            else:
                remaining_experts = self._simplify_block(nmi_encoder)
            experts.append(len(remaining_experts))
            name_experts.append(remaining_experts)
        
        return experts, name_experts
    
    def _set_weights_to_simplified_model(self, name_experts):
        print("Setting new weights to layers...")
        self._set_weights_to_new_model(name_experts)
        print("Setting new weights to experts...")
        self._set_weights_to_experts(name_experts)
        

    def simplify_original_model(self, mode='prod', name=None):
        if mode == 'prod':
            self._get_mutual_information_metrics(name)
            num_experts, name_experts = self._simplify_model()
        elif mode == 'test':
            #Simulation
            print("Simulating pruning process...")
            if name == 'qwen1.5':
                num_experts = np.random.randint(1, 60, size=24).tolist()
            elif name == 'deepseek':
                num_experts = np.random.randint(1, 64, size=26).tolist()
            elif name == 'mixtral':
                num_experts = np.random.randint(1, 8, size=32).tolist()
            else:
                print('Unsupported model for simplification')
                return None
            name_experts = []
            for n in num_experts:
                name_experts.append(list(range(0, n)))
            
        else:
            print("Incorrect mode for simplification")
            return None
        self.name_experts = name_experts
        self._build_simplified_model(num_experts, name_experts)
        self._set_weights_to_simplified_model(name_experts)
        self.simplified_model.generation_config = deepcopy(self.original_model.generation_config)
        return self.simplified_model