"""
Métricas de sostenibilidad para LLMs con poda (Transformers).
Requisitos: torch, transformers, pynvml, numpy, pandas, matplotlib.
    pip install torch transformers pynvml numpy pandas matplotlib
"""

import time
import gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from moeits.models.qwen2_moe import Qwen2MoeForCausalLM
from moeits.models.deepseek_v2_lite.modeling_deepseek import DeepseekV2ForCausalLM
from moeits.models.mixtral8x7b.modeling_mixtral import MixtralForCausalLM

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False


# --------------------------------------------------------------------------- #
# Monitor de energía / VRAM (muestreo en hilo)
# --------------------------------------------------------------------------- #
import threading

def count_experts(model) -> dict:
    """
    Recorre las capas MoE del modelo y cuenta expertos restantes por capa.
    Devuelve el total y una lista con el nº por capa.
    Funciona con arquitecturas tipo Mixtral / Qwen-MoE / DeepSeek-MoE
    donde los expertos viven en un nn.ModuleList llamado 'experts'.
    """
    per_layer = []
    for name, module in model.named_modules():
        # el contenedor típico es ...mlp.experts (un ModuleList)
        if name.endswith("experts") and hasattr(module, "__len__"):
            per_layer.append((name, len(module)))
    total_remaining = sum(n for _, n in per_layer)
    return {"per_layer": per_layer, "total_remaining": total_remaining}


def experts_pruned_pct(model, original_experts_per_layer: int, n_moe_layers: int = None):
    """
    original_experts_per_layer: nº de expertos que tenía CADA capa antes de podar
                                (p.ej. 8 en Mixtral, 64 en Qwen-MoE).
    n_moe_layers: nº de capas MoE del modelo base. Si es None se infiere
                  del propio modelo podado (asume que no se eliminaron capas enteras).
    """
    info = count_experts(model)
    n_layers = n_moe_layers if n_moe_layers is not None else len(info["per_layer"])
    total_original = original_experts_per_layer * n_layers
    pct = (1 - info["total_remaining"] / total_original) * 100.0
    return pct, info


class GPUMonitor:
    """Muestrea potencia (W) y VRAM (bytes) en un hilo aparte."""

    def __init__(self, device_index: int = 0, interval: float = 0.02):
        self.idx = device_index
        self.interval = interval
        self._stop = threading.Event()
        self.power_samples = []   # (t, watts)
        self.vram_samples = []    # bytes
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.idx) if _NVML else None

    def _loop(self):
        t0 = time.perf_counter()
        while not self._stop.is_set():
            if self._handle is not None:
                w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW -> W
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
                self.power_samples.append((time.perf_counter() - t0, w))
                self.vram_samples.append(mem)
            time.sleep(self.interval)

    def __enter__(self):
        self._stop.clear()
        self.power_samples.clear()
        self.vram_samples.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def energy_joules(self) -> float:
        """Integra potencia por trapecios -> julios."""
        if len(self.power_samples) < 2:
            return float("nan")
        t = np.array([s[0] for s in self.power_samples])
        p = np.array([s[1] for s in self.power_samples])
        return float(np.trapz(p, t))

    def peak_vram_bytes(self) -> float:
        return float(max(self.vram_samples)) if self.vram_samples else float("nan")


# --------------------------------------------------------------------------- #
# Benchmark de un modelo
# --------------------------------------------------------------------------- #
@torch.no_grad()
def benchmark_model(
    model_path: str,
    tokenizer_path: str = None,
    prompt: str = "Explain the theory of relativity in simple terms.",
    max_new_tokens: int = 256,
    n_warmup: int = 2,
    n_runs: int = 5,
    device: str = "cuda",
    dtype=torch.float16,
    original_experts_per_layer=7
) -> dict:
    if 'qwen' in model_path:
        tokenizer_path = "Qwen/Qwen1.5-MoE-A2.7B"
        model = Qwen2MoeForCausalLM.from_pretrained(model_path,device_map="auto",
                    dtype=torch.float16, 
                    trust_remote_code=True,
                    attn_implementation="eager")
    elif 'deepseek' in model_path:
        tokenizer_path = "deepseek-ai/DeepSeek-V2-Lite-Chat"
        model = DeepseekV2ForCausalLM.from_pretrained(model_path,device_map="auto",
                    dtype=torch.float16, 
                    trust_remote_code=True,
                    attn_implementation="eager")
    elif 'mixtral' in model_path:
        tokenizer_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        model = MixtralForCausalLM.from_pretrained(model_path,
                    dtype=torch.float16, 
                    trust_remote_code=True,
                    attn_implementation="eager").to(device).eval()
    else:
        tokenizer_path = tokenizer_path or model_path
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype
        ).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    inputs = tok(prompt, return_tensors="pt").to(device)

    try:
        pct, expert_info = experts_pruned_pct(model, original_experts_per_layer=original_experts_per_layer)
    except Exception:
        pct, expert_info = float("nan"), None


    def _generate_streaming():
        """Genera token a token para medir TTFT e inter-token."""
        gen_kwargs = dict(
            **inputs, max_new_tokens=1, do_sample=False,
            pad_token_id=tok.eos_token_id, use_cache=True,
        )
        # Primer token (TTFT)
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        out = model.generate(**gen_kwargs, return_dict_in_generate=True)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t_start

        past = out.past_key_values
        cur = out.sequences
        token_times = []
        for _ in range(max_new_tokens - 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step = model.generate(
                cur, past_key_values=past, max_new_tokens=1,
                do_sample=False, pad_token_id=tok.eos_token_id,
                use_cache=True, return_dict_in_generate=True,
            )
            torch.cuda.synchronize()
            token_times.append(time.perf_counter() - t0)
            past = step.past_key_values
            cur = step.sequences
        return ttft, token_times

    # Warm-up
    for _ in range(n_warmup):
        _generate_streaming()

    ttfts, inter_tokens, tps_list, energies, peak_vrams = [], [], [], [], []

    for _ in range(n_runs):
        if _NVML:
            pynvml.nvmlDeviceResetGpuLockedClocks  # no-op placeholder
        torch.cuda.reset_peak_memory_stats(device)
        with GPUMonitor(interval=0.02) as mon:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ttft, token_times = _generate_streaming()
            torch.cuda.synchronize()
            total_time = time.perf_counter() - t0

        n_generated = len(token_times) + 1
        ttfts.append(ttft)
        inter_tokens.append(np.mean(token_times) if token_times else float("nan"))
        tps_list.append(n_generated / total_time)
        energies.append(mon.energy_joules())
        # VRAM: máximo entre monitor NVML y torch (torch mide sólo el proceso)
        torch_peak = torch.cuda.max_memory_allocated(device)
        peak_vrams.append(max(mon.peak_vram_bytes(), torch_peak))

    result = {
        "model": model_path,
        "n_params": n_params,
        "vram_max_gb": np.mean(peak_vrams) / 1e9,
        "ttft_ms": np.mean(ttfts) * 1e3,
        "inter_token_ms": np.mean(inter_tokens) * 1e3,
        "tokens_per_sec": np.mean(tps_list),
        "energy_total_j": np.mean(energies),
        "j_per_token": np.mean(energies) / n_generated,
        "n_generated": n_generated,
        "experts_pruned_pct":pct,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# Ejecución sobre varios modelos / niveles de poda
# --------------------------------------------------------------------------- #
def run_suite(configs: list) -> pd.DataFrame:
    """
    configs: lista de dicts, p.ej.
        [{"name": "base",  "pruning": 0.0,  "path": "meta-llama/..."},
         {"name": "p25",   "pruning": 0.25, "path": "./ckpt_p25"},
         ...]
    'experts_pruned_pct' es opcional (para MoE / la última curva Pareto).
    """
    rows = []
    for cfg in configs:
        print(f"== Benchmarking {cfg['name']} ==")
        m = benchmark_model(cfg["path"], tokenizer_path=cfg.get("tokenizer"), original_experts_per_layer=cfg.get("num_original_experts"))
        m["name"] = cfg["name"]
        m["pruning"] = cfg.get("pruning", 0.0)
        m["experts_pruned_pct"] = m.get("experts_pruned_pct", np.nan)
        rows.append(m)
    return pd.DataFrame(rows)


def add_reduction_columns(df: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    """Reducción porcentual de cada métrica respecto al modelo baseline."""
    metrics = [
        "vram_max_gb", "ttft_ms", "inter_token_ms",
        "tokens_per_sec", "energy_total_j", "j_per_token", "n_params",
    ]
    base = df[df["name"] == baseline_name].iloc[0]
    for m in metrics:
        # reducción positiva = disminución respecto al baseline
        df[f"{m}_reduction_pct"] = (base[m] - df[m]) / base[m] * 100.0
    return df


# --------------------------------------------------------------------------- #
# Curvas de Pareto
# --------------------------------------------------------------------------- #
import matplotlib.pyplot as plt


def _pareto_front(x, y, maximize_y=True, minimize_x=True):
    """Devuelve índices no dominados. Rendimiento(y) alto, coste(x) bajo."""
    pts = list(enumerate(zip(x, y)))
    front = []
    for i, (xi, yi) in pts:
        dominated = False
        for j, (xj, yj) in pts:
            if j == i:
                continue
            better_x = (xj <= xi) if minimize_x else (xj >= xi)
            better_y = (yj >= yi) if maximize_y else (yj <= yi)
            strict = (xj != xi) or (yj != yi)
            if better_x and better_y and strict:
                dominated = True
                break
        if not dominated:
            front.append(i)
    return sorted(front, key=lambda k: x[k])


def _scatter_pareto(df, xcol, ycol, xlabel, ylabel, title,
                    perf_col="tokens_per_sec", ax=None):
    ax = ax or plt.gca()
    x = df[xcol].values
    y = df[ycol].values
    ax.scatter(x, y, s=60, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["name"], (r[xcol], r[ycol]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    front = _pareto_front(x, y, maximize_y=True, minimize_x=True)
    ax.plot(x[front], y[front], "--", color="crimson", zorder=2,
            label="Frente de Pareto")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_all_pareto(df: pd.DataFrame, perf_col="tokens_per_sec", save_prefix=None):
    """Genera las 5 curvas de Pareto solicitadas."""
    plots = [
        ("energy_total_j", perf_col, "Energía total (J)", "Rendimiento (tok/s)",
         "Rendimiento vs Energía"),
        ("vram_max_gb", perf_col, "VRAM máxima (GB)", "Rendimiento (tok/s)",
         "Rendimiento vs VRAM"),
        ("n_params", perf_col, "Nº de parámetros", "Rendimiento (tok/s)",
         "Rendimiento vs Parámetros"),
        ("inter_token_ms", perf_col, "Latencia inter-token (ms)", "Rendimiento (tok/s)",
         "Rendimiento vs Latencia"),
    ]
    for xcol, ycol, xl, yl, ti in plots:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        _scatter_pareto(df, xcol, ycol, xl, yl, ti, ax=ax)
        fig.tight_layout()
        if save_prefix:
            fig.savefig(f"{save_prefix}_{xcol}.png", dpi=200)

    # Curva especial: % expertos podados vs J/token
    sub = df.dropna(subset=["experts_pruned_pct"])
    if not sub.empty:
        sub = sub.sort_values("experts_pruned_pct")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(sub["experts_pruned_pct"], sub["j_per_token"],
                "o-", color="teal")
        for _, r in sub.iterrows():
            ax.annotate(r["name"], (r["experts_pruned_pct"], r["j_per_token"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("Expertos podados (%)")
        ax.set_ylabel("J/token")
        ax.set_title("Expertos podados vs J/token")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_prefix:
            fig.savefig(f"{save_prefix}_experts_jtoken.png", dpi=200)

    plt.show()


# --------------------------------------------------------------------------- #
# Ejemplo de uso
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    """
    print("Analyzing Mixtral-8x7B-Instruct-v0.1")
    configs = [
        {"name": "base", "pruning": 0.0, "path": "mistralai/Mixtral-8x7B-Instruct-v0.1",
         "tokenizer":"mistralai/Mixtral-8x7B-Instruct-v0.1", "num_original_experts":7},
        {"name": "f0.5-mprod", "pruning": 0.5, "path": "/MoEITS/simplified_models/mixtral_8x7b_instruct-f0.5-mprod",
         "tokenizer":"mistralai/Mixtral-8x7B-Instruct-v0.1", "num_original_experts":7},
    ]
    name = "mixtral_8x7b_instruct"

    df = run_suite(configs)
    df = add_reduction_columns(df, baseline_name="base")

    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print(df.to_string(index=False))
    df.to_csv(f"{name}_sustainability_metrics.csv", index=False)

    plot_all_pareto(df, perf_col="tokens_per_sec", save_prefix=f"{name}_pareto")

    print("Analyzing DeepSeek-V2-Lite-Chat")
    configs = [
        {"name": "base", "pruning": 0.0, "path": "deepseek-ai/DeepSeek-V2-Lite-Chat",
         "tokenizer":"deepseek-ai/DeepSeek-V2-Lite-Chat", "num_original_experts":64},
        {"name": "f1.25-mprod", "pruning": 1.25, "path": "/MoEITS/simplified_models/deepseek-v2-lite-chat-f1.25-mprod",
         "tokenizer":"deepseek-ai/DeepSeek-V2-Lite-Chat", "num_original_experts":64},
    ]
    name = "deepseek-v2-lite-chat"

    df = run_suite(configs)
    df = add_reduction_columns(df, baseline_name="base")

    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print(df.to_string(index=False))
    df.to_csv(f"{name}_sustainability_metrics.csv", index=False)

    plot_all_pareto(df, perf_col="tokens_per_sec", save_prefix=f"{name}_pareto")
    """

    print("Analyzing Qwen1.5-MoE-A2.7B-Chat")
    configs = [
        {"name": "base", "pruning": 0.0, "path": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
         "tokenizer":"Qwen/Qwen1.5-MoE-A2.7B-Chat", "num_original_experts":60},
        {"name": "f7.5-mprod", "pruning": 7.5, "path": "/MoEITS/simplified_models/qwen1.5-MoE-A2.7B-Chat-f7.5-mprod",
         "tokenizer":"Qwen/Qwen1.5-MoE-A2.7B-Chat", "num_original_experts":60},
        {"name": "f5.0-mprod", "pruning": 5.0, "path": "/MoEITS/simplified_models/qwen1.5-MoE-A2.7B-Chat-f5.0-mprod",
         "tokenizer":"Qwen/Qwen1.5-MoE-A2.7B-Chat", "num_original_experts":60},
        {"name": "f2.5-mprod", "pruning": 2.5, "path": "/MoEITS/simplified_models/qwen1.5-MoE-A2.7B-Chat-f2.5-mprod",
         "tokenizer":"Qwen/Qwen1.5-MoE-A2.7B-Chat", "num_original_experts":60},
        {"name": "f1.25-mprod", "pruning": 1.25, "path": "/MoEITS/simplified_models/qwen1.5-MoE-A2.7B-Chat-f1.25-mprod",
         "tokenizer":"Qwen/Qwen1.5-MoE-A2.7B-Chat", "num_original_experts":60},
        {"name": "f0.75-mprod", "pruning": 0.75, "path": "/MoEITS/simplified_models/qwen1.5-MoE-A2.7B-Chat-f0.75-mprod",
         "tokenizer":"Qwen/Qwen1.5-MoE-A2.7B-Chat", "num_original_experts":60},
        {"name": "f0.5-mprod", "pruning": 0.5, "path": "/MoEITS/simplified_models/qwen1.5-MoE-A2.7B-Chat-f0.5-mprod",
         "tokenizer":"Qwen/Qwen1.5-MoE-A2.7B-Chat", "num_original_experts":60},
    ]
    name = "qwen1.5-MoE-A2.7B-Chat"

    df = run_suite(configs)
    df = add_reduction_columns(df, baseline_name="base")

    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print(df.to_string(index=False))
    df.to_csv(f"{name}_sustainability_metrics.csv", index=False)

    plot_all_pareto(df, perf_col="tokens_per_sec", save_prefix=f"{name}_pareto")