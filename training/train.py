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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ------------- CONFIGURACIÓN BÁSICA -------------

MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
OUTPUT_DIR = "./models/prueba/mixtral_retrained"
USE_4BIT = False  # QLoRA (4bit) o False para bf16 LoRA
MAX_SEQ_LEN = 4096  # contexto inicial razonable
SMALL_DATASET_SAMPLES = 8000  # dataset pequeño
LARGE_DATASET_SAMPLES = 200000  # dataset grande, para la segunda fase

# Dataset de ejemplo: sustituye por uno de los generalistas que elijas.
# Aquí asumo un dataset de tipo instruction ("prompt"/"response").
DATASET_NAME = "HuggingFaceH4/instruction-dataset"
DATASET_SPLIT = "test"  # o "test" según quieras

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
            "load_in_4bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
    )
else:
    load_kwargs.update(
        {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
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


formatted = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding="max_length",
    )

tokenized = formatted.map(tokenize_fn, batched=True, remove_columns=["text"])

# Dataset pequeño (sanity check)
small_dataset = tokenized.select(range(min(SMALL_DATASET_SAMPLES, len(tokenized))))

# Dataset grande (para más adelante)
large_dataset = tokenized  # o .select(...) si quieres limitar

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ------------- CONFIGURACIÓN DEEPSPEED + TRAINER -------------

# Fichero de DeepSpeed (ejemplo ZeRO-2 para LoRA)
# Guarda esto como deepspeed_config_zero2.json en el mismo directorio.
DEEPSPEED_CONFIG_PATH = "ds_config.json"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # effective batch 16
    learning_rate=2e-4,
    num_train_epochs=1.0,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    bf16=not USE_4BIT,  # si estás en H200, usa bf16 cuando no sea 4bit
    fp16=False,
    deepspeed=DEEPSPEED_CONFIG_PATH,
    report_to="none",
    gradient_checkpointing=True,
    optim="adamw_torch",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_dataset,
    data_collator=data_collator,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Cuando el pipeline funcione, cambia `train_dataset=large_dataset`
    # y ajusta epochs/lr según tus necesidades.
