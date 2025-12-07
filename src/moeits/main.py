from moeits.Mixtral8x7b_simplification_service import Mixtral8x7b_Simplification_Service
from moeits.DeepSeekMoE_simplification_service import DeepSeekMoE_Simplification_Service
from moeits.DeepSeekV2Lite_simplification_service import DeepSeekV2Lite_Simplification_Service
from moeits.Qwen2MoE_simplification_service import Qwen2MoE_Simplification_Service
import numpy as np
import sys


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':

    model_name = sys.argv[1]
    factor = float(sys.argv[2])
    mode = sys.argv[3]
    # TODO: COMPROBACIÓN DE PARÁMETROS Y PROPAGACIÓN EN TODOS LOS MODELOS

    if 'deepseek' in model_name:
        deepseek_simp_service = DeepSeekV2Lite_Simplification_Service("deepseek-ai/DeepSeek-V2-Lite", factor=factor)
        simplified_deepseek_moe = deepseek_simp_service.simplify_original_model()
        print(f"Params: {count_trainable_parameters(simplified_deepseek_moe)}")
        simplified_deepseek_moe.save_pretrained(f'/MoEITS/simplified_models/deepseek-v2-lite-f{factor}/')
    elif 'mixtral' in model_name:
        mixtral_simp_service = Mixtral8x7b_Simplification_Service('"mistralai/Mixtral-8x7B-Instruct-v0.1"', factor=factor)
        simplified_mixtral = mixtral_simp_service.simplify_original_model()
        print(f"Params: {count_trainable_parameters(simplified_mixtral)}")
        simplified_mixtral.save_pretrained(f'/MoEITS/simplified_models/mixtral_8x7b_instruct-f{factor}/')
    elif 'qwen' in model_name:
        qwen_simp_service = Qwen2MoE_Simplification_Service("Qwen/Qwen1.5-MoE-A2.7B", factor=factor)
        simplified_qwen = qwen_simp_service.simplify_original_model(mode=mode, name=model_name)
        print(f"Params: {count_trainable_parameters(simplified_qwen)}")
        simplified_qwen.save_pretrained(f'/MoEITS/simplified_models/qwen2-moe-f{factor}-m{mode}/')

