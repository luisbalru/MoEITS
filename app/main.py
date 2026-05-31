from moeits.Mixtral8x7b_simplification_service import Mixtral8x7b_Simplification_Service
from moeits.DeepSeekMoE_simplification_service import DeepSeekMoE_Simplification_Service
from moeits.DeepSeekV2Lite_simplification_service import DeepSeekV2Lite_Simplification_Service
from moeits.Qwen2MoE_simplification_service import Qwen2MoE_Simplification_Service
from moeits.Qwen3_5_simplification_service import Qwen3_5_Simplification_Service
import numpy as np
import sys
import argparse



def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MoEITS setup")

    parser.add_argument(
        "--model-name",
        choices=["deepseekv2lite", "mixtral8x7B", "qwen1.5", "Qwen3.5-35B-A3B"],
        required=True
    )

    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        required=True
    )

    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--factor", type=float, default=None)


    args = parser.parse_args()

    model_name = args.model_name
    factor = args.factor
    num_experts = args.num_experts
    mode = args.mode

    if factor == None and num_experts == None:
        print("Error. Both factor and number of experts cannot be None. Exiting")
        exit()

    if 'deepseekv2lite' in model_name:
        deepseek_simp_service = DeepSeekV2Lite_Simplification_Service("deepseek-ai/DeepSeek-V2-Lite-Chat", factor=factor, number_of_experts=num_experts)
        simplified_deepseek_moe = deepseek_simp_service.simplify_original_model(mode=mode, name=model_name)
        print(f"Params: {count_trainable_parameters(simplified_deepseek_moe)}")
        if num_experts:
            simplified_deepseek_moe.save_pretrained(f'/MoEITS/simplified_models/deepseek-v2-lite-chat-ne{num_experts}-m{mode}/')
        else:
            simplified_deepseek_moe.save_pretrained(f'/MoEITS/simplified_models/deepseek-v2-lite-chat-f{factor}-m{mode}/')
    elif 'mixtral8x7B' in model_name:
        mixtral_simp_service = Mixtral8x7b_Simplification_Service("mistralai/Mixtral-8x7B-Instruct-v0.1", factor=factor, number_of_experts=num_experts)
        simplified_mixtral = mixtral_simp_service.simplify_original_model(mode=mode, name=model_name)
        print(f"Params: {count_trainable_parameters(simplified_mixtral)}")
        if num_experts:
            simplified_mixtral.save_pretrained(f'/MoEITS/simplified_models/mixtral_8x7b_instruct-ne{num_experts}-m{mode}/')
        else:
            simplified_mixtral.save_pretrained(f'/MoEITS/simplified_models/mixtral_8x7b_instruct-f{factor}-m{mode}/')
    elif 'qwen1.5' in model_name:
        qwen_simp_service = Qwen2MoE_Simplification_Service("Qwen/Qwen1.5-MoE-A2.7B-Chat", factor=factor, number_of_experts=num_experts)
        simplified_qwen = qwen_simp_service.simplify_original_model(mode=mode, name=model_name)
        print(f"Params: {count_trainable_parameters(simplified_qwen)}")
        if num_experts:
            simplified_qwen.save_pretrained(f'/MoEITS/simplified_models/qwen1.5-MoE-A2.7B-Chat-ne{num_experts}-m{mode}/')
        else:    
            simplified_qwen.save_pretrained(f'/MoEITS/simplified_models/qwen1.5-MoE-A2.7B-Chat-f{factor}-m{mode}/')
    elif "Qwen3.5-35B-A3B" in model_name:
        qwen_simp_service = Qwen3_5_Simplification_Service("Qwen3.5-35B-A3B", number_of_experts=num_experts)
        simplified_qwen = qwen_simp_service.simplify_original_model(mode=mode, name=model_name)


