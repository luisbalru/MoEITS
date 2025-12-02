import json
import os
import gc
import torch
from MoEITS_evaluation import MoEITSEvaluation

# ---------------- CONFIGURATION ----------------
MODEL_PATH = "Qwen/Qwen1.5-MoE-A2.7B" 
OUTPUT_FILE = "Qwen1.5-MoE-A2.7B_benchmark_results.json"
FAIL_LOG = "benchmark_failures.log"

# Ensure API Key is present for benchmarks that use LLM-Evals (TruthfulQA, IFEval)
if "OPENAI_API_KEY" not in os.environ:
    print("⚠️ WARNING: OPENAI_API_KEY not found. Benchmarks using LLM-as-a-judge (TruthfulQA, IFEval) may fail or score poorly.")
# -----------------------------------------------

# Import Benchmarks
from deepeval.benchmarks import (
    MMLU, HellaSwag, BigBenchHard, TruthfulQA, HumanEval,
    IFEval, GSM8K, MathQA, LogiQA, BoolQ, ARC, BBQ, 
    LAMBADA, Winogrande, SQuAD # Fixed lambada -> LAMBADA
)

def save_result(name, score, results_dict):
    """Saves result immediately to JSON."""
    print(f"✅ {name}: {score}")
    results_dict[name] = score
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results_dict, f, indent=4)

def log_failure(name, error_msg):
    """Logs crashes to a separate file without stopping the suite."""
    print(f"❌ {name} FAILED: {error_msg}")
    with open(FAIL_LOG, "a") as f:
        f.write(f"{name}: {error_msg}\n")

def clear_cache():
    """Forces GPU memory cleanup between heavy benchmarks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_benchmark_safe(name, benchmark_cls, model, results_dict, **kwargs):
    """Wrapper to run a benchmark with error handling."""
    if name in results_dict:
        print(f"⏭️  Skipping {name} (Already computed: {results_dict[name]})")
        return

    print(f"\nrunning {name}...")
    try:
        # Instantiate benchmark with arguments
        bench = benchmark_cls(**kwargs)
        # Run evaluation
        # Note: Some benchmarks allow batch_size in evaluate, others don't.
        # We handle this by checking kwargs for batch_size or passing it if valid.
        eval_kwargs = {}
        if 'batch_size' in kwargs:
            # Separate init args from eval args if necessary, 
            # but usually DeepEval takes batch_size in evaluate()
            pass 
        
        # Determine batch size logic (simplification)
        b_size = kwargs.pop('batch_size', 16) 
        
        # Execute
        bench.evaluate(model=model, batch_size=b_size)
        save_result(name, bench.overall_score, results_dict)
        
    except Exception as e:
        log_failure(name, str(e))
    finally:
        clear_cache()

def run_suite():
    # 1. Initialize Wrapper
    print(f"Loading Model: {MODEL_PATH}")
    moe_model = MoEITSEvaluation(model_path=MODEL_PATH)
    
    results = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ JSON corrupted, starting fresh.")
                results = {}

    # --- SUITE 1: Knowledge & Common Sense ---
    print("\n--- Starting SUITE 1: Knowledge ---")
    
    run_benchmark_safe("MMLU", MMLU, moe_model, results, n_shots=5, batch_size=16)
    run_benchmark_safe("HellaSwag", HellaSwag, moe_model, results, n_shots=10, batch_size=16)
    run_benchmark_safe("ARC", ARC, moe_model, results, n_shots=25, batch_size=16)
    run_benchmark_safe("Winogrande", Winogrande, moe_model, results, n_shots=5, batch_size=16)
    run_benchmark_safe("BoolQ", BoolQ, moe_model, results, n_shots=0, batch_size=32)
    run_benchmark_safe("LAMBADA", LAMBADA, moe_model, results, n_shots=0, batch_size=32)

    # --- SUITE 2: Hard Reasoning ---
    print("\n--- Starting SUITE 2: Reasoning ---")

    # Lower batch size for reasoning tasks to prevent OOM
    run_benchmark_safe("GSM8K", GSM8K, moe_model, results, n_shots=3, enable_cot=True, batch_size=8)
    run_benchmark_safe("MathQA", MathQA, moe_model, results, n_shots=5, batch_size=16)
    run_benchmark_safe("LogiQA", LogiQA, moe_model, results, n_shots=0, batch_size=16)
    run_benchmark_safe("BigBenchHard", BigBenchHard, moe_model, results, n_shots=3, batch_size=8)

    # --- SUITE 3: Specialist (Slow/Expensive) ---
    print("\n--- Starting SUITE 3: Specialist ---")

    # HumanEval: k=1 is the metric, n=10 is samples generated. 
    # Warning: This is very slow.
    run_benchmark_safe("HumanEval", HumanEval, moe_model, results, n=10, batch_size=1) 
    
    run_benchmark_safe("TruthfulQA", TruthfulQA, moe_model, results, batch_size=16)
    run_benchmark_safe("IFEval", IFEval, moe_model, results, batch_size=16)
    run_benchmark_safe("SQuAD", SQuAD, moe_model, results, n_shots=3, batch_size=16)
    run_benchmark_safe("BBQ", BBQ, moe_model, results, batch_size=16)

    print("\n🎉 Benchmarking Complete!")

if __name__ == "__main__":
    run_suite()