from Mixtral8x7b_simplification_service import Mixtral8x7b_Simplification_Service
from DeepSeekMoE_simplification_service import DeepSeekMoE_Simplification_Service
import numpy as np
import sys


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':
    model_name = sys.argv[1]
    factor = float(sys.argv[2])

    if 'deepseek' in model_name:
        deepseek_simp_service = DeepSeekMoE_Simplification_Service('deepseek-ai/deepseek-moe-16b-base', factor=factor)
        simplified_deepseek_moe = deepseek_simp_service.simplify_original_model()
        print(f"Params: {count_trainable_parameters(simplified_deepseek_moe)}")
        simplified_deepseek_moe.save_pretrained(f'simplified_models/deepseek-moe-16b-f{factor}/')
    elif 'mixtral' in model_name:
        mixtral_simp_service = Mixtral8x7b_Simplification_Service('mistralai/Mixtral-8x7B-v0.1', factor=factor)
        simplified_mixtral = mixtral_simp_service.simplify_original_model()
        print(f"Params: {count_trainable_parameters(simplified_mixtral)}")
        simplified_mixtral.save_pretrained(f'simplified_models/mixtral_8x7b-f{factor}/')
