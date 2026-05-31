import numpy as np
from scipy.stats import entropy, iqr
import pandas as pd
import os
from numpy import unravel_index
import json
from abc import ABC, abstractmethod
from copy import deepcopy
import torch

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
    
    def _simplify_block_fixed_number_of_experts_numpy(self, nmi_matrix):
        remaining_experts = np.arange(nmi_matrix.shape[0]).tolist()
        while len(remaining_experts) > self.number_of_experts:
            arg_e1, arg_e2 = unravel_index(np.argmax(nmi_matrix), nmi_matrix.shape)
            closeness_e1 = np.mean(nmi_matrix[arg_e1,:])
            closeness_e2 = np.mean(nmi_matrix[arg_e2,:])
            arg_expert_remove = arg_e1 if closeness_e1 > closeness_e2 else arg_e2
            remaining_experts.pop(arg_expert_remove)
            nmi_matrix = np.delete(np.delete(nmi_matrix, arg_expert_remove, axis=0), arg_expert_remove, axis=1)
        return remaining_experts
    
    def _simplify_block_fixed_number_of_experts(self, nmi_matrix):
        """
        PyTorch version of the expert pruning method.
        Expects nmi_matrix to be a 2D PyTorch Tensor.
        """
        device = nmi_matrix.device
        num_experts = nmi_matrix.shape[0]
        
        # Track original indices using a 1D tensor
        remaining_experts = torch.arange(num_experts, device=device)
        
        while remaining_experts.numel() > self.number_of_experts:
            # 1. Find the flat index of the maximum value
            flat_arg = torch.argmax(nmi_matrix)
            
            # 2. Unravel the 1D index into 2D coordinates (row, col)
            cols = nmi_matrix.shape[1]
            arg_e1 = flat_arg // cols
            arg_e2 = flat_arg % cols
            
            # 3. Calculate mean closeness for both candidate experts
            closeness_e1 = torch.mean(nmi_matrix[arg_e1, :])
            closeness_e2 = torch.mean(nmi_matrix[arg_e2, :])
            
            # 4. Identify which expert is more redundant (higher mean closeness)
            arg_expert_remove = arg_e1 if closeness_e1 > closeness_e2 else arg_e2
            
            # 5. Create a boolean mask to filter out the removed expert
            # This replaces np.delete for both the list and the matrix rows/cols
            mask = torch.ones(nmi_matrix.shape[0], dtype=torch.bool, device=device)
            mask[arg_expert_remove] = False
            
            # 6. Shrink the tracking tensor and the matrix simultaneously
            remaining_experts = remaining_experts[mask]
            nmi_matrix = nmi_matrix[mask][:, mask]
        
        return remaining_experts.tolist()

    def _simplify_model(self):
        experts = []
        name_experts = []

        print("Simplifying model...")
        for k in self.layers.keys():
            nmi_encoder = self.layers[k]
            if self.number_of_experts:
                remaining_experts = self._simplify_block_fixed_number_of_experts(torch.from_numpy(nmi_encoder))
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
        

    def simplify_original_model(self, mode='online', name=None):        
        self._get_mutual_information_metrics(name)
        num_experts, name_experts = self._simplify_model()
        self.name_experts = name_experts

        if mode == 'online':
            self._build_simplified_model(num_experts, name_experts)
            self._set_weights_to_simplified_model(name_experts)
            self.simplified_model.generation_config = deepcopy(self.original_model.generation_config)
            return self.simplified_model
        elif mode == "offline":
            print(self.name_experts)
        
        else:
            print("Unknown mode")