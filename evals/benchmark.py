import json
import os
import gc
import torch
from MoEITS_evaluation import MoEITSEvaluation

# Import Benchmarks (Case Sensitive Fixes)
from deepeval.benchmarks import (
    MMLU, HellaSwag, BigBenchHard, TruthfulQA, HumanEval,
    IFEval, GSM8K, MathQA, LogiQA, BoolQ, ARC, BBQ, 
    LAMBADA, Winogrande, SQuAD 
)
from deepeval.benchmarks.modes import TruthfulQAMode # For specific config

# ---------------- CONFIGURATION ----------------
OUTPUT_FILE = "Qwen1.5-MoE_results.json"
FAIL_LOG = "failures.log"
# -----------------------------------------------

def save_result(name, score, results_dict):
    print(f"✅ {name}: {score}")
    results_dict[name] = score
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results_dict, f, indent=4)

def log_failure(name, error_msg):
    print(f"❌ {name} FAILED: {error_msg}")
    with open(FAIL_LOG, "a") as f:
        f.write(f"{name}: {error_msg}\n")

def clear_cache():
    """Aggressive memory cleanup between benchmarks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_benchmark(name, benchmark_obj, model, results_dict, **eval_kwargs):
    """Generic runner with error handling."""
    if name in results_dict:
        print(f"⏭️  Skipping {name} (Already computed)")
        return

    print(f"\n🚀 Running {name}...")
    try:
        benchmark_obj.evaluate(model=model, **eval_kwargs)
        save_result(name, benchmark_obj.overall_score, results_dict)
    except Exception as e:
        log_failure(name, str(e))
    finally:
        clear_cache()

def run_suite():
    # 1. Load Model Once
    moe_model = MoEITSEvaluation(model_path="mistralai/Mixtral-8x7B-Instruct-v0.1")
    
    results = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            try: results = json.load(f)
            except: pass

    # --- SUITE 1: Knowledge (Fast / Low Memory) ---
    print("\n--- SUITE 1: Knowledge ---")
    
    run_benchmark("MMLU", MMLU(n_shots=5), moe_model, results, batch_size=16)
    run_benchmark("HellaSwag", HellaSwag(n_shots=5), moe_model, results, batch_size=16)
    run_benchmark("ARC", ARC(n_shots=5), moe_model, results, batch_size=16)
    run_benchmark("BoolQ", BoolQ(n_shots=0), moe_model, results, batch_size=32)
    run_benchmark("Winogrande", Winogrande(n_shots=5), moe_model, results, batch_size=16)
    run_benchmark("LAMBADA", LAMBADA(n_shots=0), moe_model, results, batch_size=32)

    # --- SUITE 2: Reasoning (Computationally Heavy) ---
    print("\n--- SUITE 2: Reasoning ---")
    
    # GSM8K: Uses exact match on numbers.
    run_benchmark("GSM8K", GSM8K(n_shots=3, enable_cot=True), moe_model, results, batch_size=4)
    
    run_benchmark("MathQA", MathQA(n_shots=3), moe_model, results, batch_size=8)
    run_benchmark("LogiQA", LogiQA(n_shots=0), moe_model, results, batch_size=8)
    run_benchmark("BigBenchHard", BigBenchHard(n_shots=3), moe_model, results, batch_size=4)

    # --- SUITE 3: Specialist (Complex / Slow) ---
    print("\n--- SUITE 3: Specialist ---")

    # TruthfulQA: MC1 mode is deterministic and doesn't require an LLM Judge (save API costs)
    run_benchmark("TruthfulQA", TruthfulQA(mode=TruthfulQAMode.MC1), moe_model, results, batch_size=16)

    # HumanEval: Very slow. Generates code.
    # We use n=10 samples per problem to check pass@1 (requires generate_samples in wrapper)
    run_benchmark("HumanEval", HumanEval(n=10), moe_model, results, k=1)

    # NOTE: The following often require an OPENAI_API_KEY for the 'Judge' model.
    # If you don't have one, these may fail or output 0.
    if os.getenv("OPENAI_API_KEY"):
        run_benchmark("IFEval", IFEval(), moe_model, results)
        run_benchmark("SQuAD", SQuAD(n_shots=3), moe_model, results)
        run_benchmark("BBQ", BBQ(), moe_model, results)
    else:
        print("\n⚠️  Skipping IFEval, SQuAD, BBQ (Require OPENAI_API_KEY for grading)")

if __name__ == "__main__":
    run_suite()