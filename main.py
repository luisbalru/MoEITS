from Mixtral8x7b_simplification_service import Mixtral8x7b_Simplification_Service
from DeepSeekMoE_simplification_service import DeepSeekMoE_Simplification_Service
import numpy as np


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':
    deepseek_simp_service = DeepSeekMoE_Simplification_Service('deepseek-ai/deepseek-moe-16b-base')
    simplified_deepseek_moe = deepseek_simp_service.simplify_original_model()
    print(simplified_deepseek_moe)
    print(f"Params: {count_trainable_parameters(simplified_deepseek_moe)}")
    simplified_deepseek_moe.save_pretrained('simplified_models/deepseek-moe-16b/')