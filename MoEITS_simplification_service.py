import numpy as np
from scipy.stats import entropy, iqr
import pandas as pd
from transformers import AutoTokenizer, MixtralForCausalLM, MixtralConfig, AutoModelForCausalLM
import os
from numpy import unravel_index
import torch
import json


class MoEITS_Simplification_Service:

    def __init__(self, model_name, output_base_path='', config_path='utils/config.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.model_name = model_name
        self.output_base_path = output_base_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=config['token'])
        self.original_model = AutoModelForCausalLM.from_pretrained(self.model_name, token=config['token'], trust_remote_code=True, dtype="auto")
        self.simplified_model = None
        self.layers = {}


