from MoEITS_simplification_service import MoEITS_Simplification_Service
import numpy as np

moe_simp_service = MoEITS_Simplification_Service("deepseek-ai/deepseek-moe-16b-base")
print("Parámetros modelo original: ", np.sum(p.numel() for p in moe_simp_service.original_model.parameters() if p.requires_grad))
input()
print(moe_simp_service.original_model.layers[1].mlp)