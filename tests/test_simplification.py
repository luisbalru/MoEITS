from moeits.DeepSeekV2Lite_simplification_service import DeepSeekV2Lite_Simplification_Service
from moeits.Mixtral8x7b_simplification_service import Mixtral8x7b_Simplification_Service
from moeits.Qwen2MoE_simplification_service import Qwen2MoE_Simplification_Service


def test_qwen_simplification_service():
    factor = 5
    simp_service = Qwen2MoE_Simplification_Service("Qwen/Qwen1.5-MoE-A2.7B", factor=factor)
    simp_model = simp_service.simplify_original_model(mode='test', name = "qwen")
    num_layers = len(simp_service.name_experts) == len(simp_model.model.layers)
    assert num_layers