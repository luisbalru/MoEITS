#pip install flash-attn --no-build-isolation
#export OPENAI_API_KEY="sk-..."

import json
import os
from MoEITS_evaluation import MoEITSEvaluation

# Import all 15 Benchmarks
from deepeval.benchmarks import (
    MMLU, HellaSwag, BigBenchHard, TruthfulQA, HumanEval,
    IFEval, GSM8K, MathQA, LogiQA, BoolQ, ARC, BBQ, 
    Lambada, Winogrande, SQuAD
)

# ---------------- CONFIGURATION ----------------
MODEL_PATH = "" 
OUTPUT_FILE = "moe_benchmark_results.json"
# -----------------------------------------------

def save_result(name, score, results_dict):
    print(f"✅ {name}: {score}")
    results_dict[name] = score
    # Save incrementally so you don't lose progress if it crashes
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results_dict, f, indent=4)

def run_suite():
    # 1. Initialize Wrapper
    moe_model = MoEITSEvaluation(model_path=MODEL_PATH)
    
    results = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            results = json.load(f)

    # --- SUITE 1: Knowledge & Common Sense (Deterministic / Fast) ---
    print("\n--- Starting SUITE 1: Knowledge ---")
    
    # MMLU (General Knowledge)
    if "MMLU" not in results:
        bench = MMLU(n_shots=5)
        bench.evaluate(model=moe_model, batch_size=16)
        save_result("MMLU", bench.overall_score, results)

    # HellaSwag (Common Sense)
    if "HellaSwag" not in results:
        bench = HellaSwag(n_shots=10)
        bench.evaluate(model=moe_model, batch_size=16)
        save_result("HellaSwag", bench.overall_score, results)

    # ARC (Reasoning Challenge)
    if "ARC" not in results:
        bench = ARC(n_shots=25)
        bench.evaluate(model=moe_model, batch_size=16)
        save_result("ARC", bench.overall_score, results)

    # Winogrande (Pronoun Resolution)
    if "Winogrande" not in results:
        bench = Winogrande(n_shots=5)
        bench.evaluate(model=moe_model, batch_size=16)
        save_result("Winogrande", bench.overall_score, results)

    # BoolQ (Yes/No)
    if "BoolQ" not in results:
        bench = BoolQ(n_shots=0)
        bench.evaluate(model=moe_model, batch_size=32)
        save_result("BoolQ", bench.overall_score, results)

    # LAMBADA (Word Prediction)
    if "Lambada" not in results:
        bench = Lambada(n_shots=0)
        bench.evaluate(model=moe_model, batch_size=32)
        save_result("Lambada", bench.overall_score, results)

    # --- SUITE 2: Hard Reasoning (Math & Logic) ---
    print("\n--- Starting SUITE 2: Reasoning ---")

    # GSM8K (Grade School Math)
    if "GSM8K" not in results:
        bench = GSM8K(n_shots=3, enable_cot=True)
        bench.evaluate(model=moe_model, batch_size=8) # Lower batch size for CoT
        save_result("GSM8K", bench.overall_score, results)

    # MathQA (Complex Math)
    if "MathQA" not in results:
        bench = MathQA(n_shots=5)
        bench.evaluate(model=moe_model, batch_size=16)
        save_result("MathQA", bench.overall_score, results)

    # LogiQA (Logic Puzzles)
    if "LogiQA" not in results:
        bench = LogiQA(n_shots=0)
        bench.evaluate(model=moe_model, batch_size=16)
        save_result("LogiQA", bench.overall_score, results)

    # BigBench Hard (Challenging Tasks)
    if "BigBenchHard" not in results:
        bench = BigBenchHard(n_shots=3)
        bench.evaluate(model=moe_model, batch_size=8)
        save_result("BigBenchHard", bench.overall_score, results)

    # --- SUITE 3: Safety, Instruction & Code (Expensive / Slow) ---
    print("\n--- Starting SUITE 3: Specialist ---")

    # HumanEval (Coding - Very Slow due to sampling)
    if "HumanEval" not in results:
        bench = HumanEval(n=10) # n=10 samples per problem
        bench.evaluate(model=moe_model, k=1) # Check pass@1
        save_result("HumanEval", bench.overall_score, results)

    # TruthfulQA (Hallucinations)
    if "TruthfulQA" not in results:
        bench = TruthfulQA()
        bench.evaluate(model=moe_model, batch_size=16)
        save_result("TruthfulQA", bench.overall_score, results)

    # IFEval (Instruction Following)
    if "IFEval" not in results:
        bench = IFEval()
        bench.evaluate(model=moe_model, batch_size=16)
        save_result("IFEval", bench.overall_score, results)
        
    # SQuAD (Reading Comprehension)
    if "SQuAD" not in results:
        bench = SQuAD(n_shots=3)
        bench.evaluate(model=moe_model) # Batching often disabled for SQuAD logic
        save_result("SQuAD", bench.overall_score, results)

    # BBQ (Bias)
    if "BBQ" not in results:
        bench = BBQ()
        bench.evaluate(model=moe_model, batch_size=16)
        save_result("BBQ", bench.overall_score, results)

if __name__ == "__main__":
    run_suite()