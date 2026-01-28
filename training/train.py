# train_mixtral_lora_deepspeed.py

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ------------- CONFIGURACIÓN BÁSICA -------------

MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
OUTPUT_DIR = "./models/prueba/mixtral_retrained"
USE_4BIT = True  # QLoRA (4bit) o False para bf16 LoRA
MAX_SEQ_LEN = 1024  # contexto inicial razonable
SMALL_DATASET_SAMPLES = 8000  # dataset pequeño
LARGE_DATASET_SAMPLES = 500000  # dataset grande, para la segunda fase

# Dataset de ejemplo: sustituye por uno de los generalistas que elijas.
# Aquí asumo un dataset de tipo instruction ("prompt"/"response").
DATASET_NAME = "teknium/OpenHermes-2.5"
DATASET_SPLIT = "train"

# ------------- CARGA DE TOKENIZER Y MODELO -------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    padding_side="right",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

load_kwargs = {}
if USE_4BIT:
    load_kwargs.update(
        {
            "device_map": "auto",
            "dtype": torch.float16,
            "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,                    # ← AQUÍ SÍ
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
        }
    )
else:
    load_kwargs.update(
        {
            "device_map": "auto",
            "dtype": torch.bfloat16,
        }
    )

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)

if USE_4BIT:
    model = prepare_model_for_kbit_training(model)

# Config LoRA: ajusta `target_modules` a los nombres reales en Mixtral
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate",
        "w1",
        "w2",
        "w3"
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ------------- PREPARACIÓN DEL DATASET -------------

# Cargamos un dataset generalista; en la práctica, selecciona uno concreto
raw_dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

# Función de formateo: ajusta según los campos reales del dataset
def format_example(example):
    prompt = example.get("prompt") or ""
    input_text = example.get("input") or ""
    completion = example.get("completion") or ""

    # Plantilla sencilla estilo instruct
    if input_text:
        full_prompt = (
            "Instruction:\n"
            f"{prompt}\n\n"
            "Input:\n"
            f"{input_text}\n\n"
            "Response:\n"
        )
    else:
        full_prompt = (
            "Instruction:\n"
            f"{prompt}\n\n"
            "Response:\n"
        )

    full_text = full_prompt + completion + tokenizer.eos_token
    return {"text": full_text}

def format_openhermes(example):
    """OpenHermes usa 'instruction', 'input', 'output'"""
    instruction = example["conversations"][0]['value']
    input_text = example.get("input") or ""
    output = example["conversations"][1]['value']
    
    if input_text:
        text = f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nResponse:\n{output}{tokenizer.eos_token}"
    else:
        text = f"Instruction:\n{instruction}\n\nResponse:\n{output}{tokenizer.eos_token}"
    return {"text": text}


formatted = raw_dataset.map(format_openhermes, remove_columns=raw_dataset.column_names)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding="max_length",
    )

tokenized = formatted.map(tokenize_fn, batched=True, remove_columns=["text"])

# Dataset pequeño (sanity check)
small_dataset = tokenized.select(range(min(LARGE_DATASET_SAMPLES, len(tokenized))))

# Dataset grande (para más adelante)
large_dataset = tokenized  # o .select(...) si quieres limitar

from dataclasses import dataclass
from typing import Any, Dict, List

def data_collator(features):
    """Collator minimalista PROBADO con DeepSpeed + Mixtral"""
    max_len = max(len(f["input_ids"]) for f in features)
    
    batch = {
        "input_ids": torch.stack([torch.nn.functional.pad(
            torch.tensor(f["input_ids"], dtype=torch.long), 
            (0, max_len - len(f["input_ids"]))
        ) for f in features]),
        "attention_mask": torch.stack([torch.nn.functional.pad(
            torch.tensor(f["attention_mask"], dtype=torch.long), 
            (0, max_len - len(f["attention_mask"]))
        ) for f in features]),
    }
    
    # Labels con -100 donde attention_mask=0
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    
    return batch

data_collator = data_collator

# ------------- CONFIGURACIÓN DEEPSPEED + TRAINER -------------

# Fichero de DeepSpeed (ejemplo ZeRO-2 para LoRA)
# Guarda esto como deepspeed_config_zero2.json en el mismo directorio.
DEEPSPEED_CONFIG_PATH = "ds_config.json"

# ← TrainingArguments corregidos
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    
    learning_rate=2e-4,
    num_train_epochs=3,
    max_steps=20,  # ← test corto
    
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    
    bf16=True,  # ← bf16 para QLoRA + H200
    fp16=False,  # ← False
    
    deepspeed="ds_config.json",
    gradient_checkpointing=False,
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

# ← Explícito antes del Trainer
model.config.use_cache = False
model.gradient_checkpointing_disable()  # explícito

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=large_dataset,
    data_collator=data_collator,
)

if __name__ == "__main__":
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Cuando el pipeline funcione, cambia `train_dataset=large_dataset`
    # y ajusta epochs/lr según tus necesidades.
