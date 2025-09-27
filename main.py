#from MoEITS_simplification_service import MoEITS_Simplification_Service
from Mixtral8x7b_simplification_service import Mixtral8x7b_Simplification_Service
import numpy as np

moe_simp_service = Mixtral8x7b_Simplification_Service("mistralai/Mixtral-8x7B-Instruct-v0.1")
print("Parámetros modelo original: ", np.sum(p.numel() for p in moe_simp_service.original_model.parameters() if p.requires_grad))
print(moe_simp_service.simplify_original_model())