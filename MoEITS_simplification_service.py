import numpy as np
from scipy.stats import entropy, iqr
import pandas as pd
from transformers import AutoTokenizer, MixtralForCausalLM, MixtralConfig, AutoModelForCausalLM
import os
from numpy import unravel_index
import torch
import json
from abc import ABC, abstractmethod


class MoEITS_Simplification_Service(ABC):

    def __init__(self, model_name, output_base_path='', config_path='utils/config.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.model_name = model_name
        self.output_base_path = output_base_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=config['token'])
        self.original_model = AutoModelForCausalLM.from_pretrained(self.model_name, token=config['token'], trust_remote_code=True, dtype="auto")
        self.simplified_model = None
        self.layers = {}

    @abstractmethod
    def _get_mutual_information_metrics(self):
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


    def _simplify_block(self, nmi_matrix):
        remaining_experts = list(np.arange(nmi_matrix.shape[0]))
        mean = np.mean(nmi_matrix)
        iqr_val = iqr(nmi_matrix)
        th = mean + iqr_val*1.5
        while len(np.where(nmi_matrix > th)[0]):
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
            remaining_experts = self._simplify_block(nmi_encoder)
            experts.append(len(remaining_experts))
            name_experts.append(remaining_experts)
        
        return experts, name_experts
    
    def _set_weights_to_simplified_model(self, name_experts):
        print("Setting new weights to layers...")
        self._set_weights_to_new_model(name_experts)
        print("Setting new weights to experts...")
        self._set_weights_to_experts(name_experts)

    def simplified_original_model(self):
        self._get_mutual_information_metrics()
        num_experts, name_experts = self._simplify_model()
        self._build_simplified_model(num_experts, name_experts)
        self._set_weights_to_simplified_model(name_experts)
        return self.simplified_model