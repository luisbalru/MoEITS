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
import gc
from moeits.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training+
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ------------- CONFIGURACIÓN BÁSICA -------------

model_name = sys.argv[1]

if 'qwen' in model_name:
    TOKENIZER_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
elif 'deepseek' in model_name:
    TOKENIZER_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"
elif 'mixtral' in model_name:
    TOKENIZER_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

USE_4BIT = False  # QLoRA (4bit) o False para bf16 LoRA
MAX_SEQ_LEN = 2048  # contexto inicial razonable
SMALL_DATASET_SAMPLES = 8000  # dataset pequeño
LARGE_DATASET_SAMPLES = 500000  # dataset grande, para la segunda fase

# Dataset de ejemplo: sustituye por uno de los generalistas que elijas.
# Aquí asumo un dataset de tipo instruction ("prompt"/"response").
DATASET_NAME = "teknium/OpenHermes-2.5"
DATASET_SPLIT = "train"
DEEPSPEED_CONFIG_PATH = "ds_config.json"


def train(model_name, output_dir):
    # ------------- CARGA DE TOKENIZER Y MODELO -------------

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}
    if USE_4BIT:
        load_kwargs.update(
            {
                #"device_map": "auto",
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
                #"device_map": "auto",
                "dtype": torch.bfloat16,
                "attn_implementation":"sdpa"
            }
        )

    #model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)

    model = Qwen2MoeForCausalLM.from_pretrained(model_name, **load_kwargs)


    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    """
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
    """
    # Config LoRA: ajusta `target_modules` a los nombres reales en Qwen
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
            "gate_proj",
            "up_proj",
            "down_proj",
            "gate" # El router del MoE
        ],
    )


    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.config.use_cache = False

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
        )

    tokenized = formatted.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Dataset pequeño (sanity check)
    #small_dataset = tokenized.select(range(min(LARGE_DATASET_SAMPLES, len(tokenized))))

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

    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # ------------- CONFIGURACIÓN DEEPSPEED + TRAINER -------------

    # Fichero de DeepSpeed (ejemplo ZeRO-2 para LoRA)
    # Guarda esto como deepspeed_config_zero2.json en el mismo directorio

    # ← TrainingArguments corregidos
    
    #model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=output_dir,
        
        per_device_train_batch_size=16,   
        gradient_accumulation_steps=4,
        
        learning_rate=2e-4,
        max_steps=900,
        
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        
        bf16=True,
        #deepspeed=DEEPSPEED_CONFIG_PATH,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        
        lr_scheduler_type="cosine",      # Curva de aprendizaje natural
        warmup_ratio=0.05,               # Dedica el 5% del tiempo a calentar motores suavemente
        weight_decay=0.05,               # Regularización masiva para evitar memorizar (generaliza mejor)
        neftune_noise_alpha=5.0,         # Magia negra para subir puntos en HellaSwag/ARC
        optim="adamw_torch",             # Asegura el optimizador correcto para weight_decay
        # ----------------------------------------
    )



    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=64,   # effective batch 64
        learning_rate=2e-4,
        num_train_epochs=2,               # y sin max_steps
        logging_steps=20,
        max_steps=1500,
        save_steps=1000,
        save_total_limit=3,
        bf16=True,
        fp16=False,
        deepspeed=DEEPSPEED_CONFIG_PATH,
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="adamw_torch",
    )
    """
    # ← Explícito antes del Trainer
    model.config.use_cache = False
    #model.gradient_checkpointing_disable()  # explícito

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=large_dataset,
        data_collator=data_collator,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    #models = ["/MoEITS/simplified_models/qwen1.5-MoE-A2.7B-Chat-f2.5-mprod", "/MoEITS/simplified_models/qwen1.5-MoE-A2.7B-Chat-f5.0-mprod"]
    #output_dirs = ["./models/prueba/qwen1.5-MoE-A2.7B-Chat-f2.5-mprod_retrained", "./models/prueba/qwen1.5-MoE-A2.7B-Chat-f5.0-mprod_retrained"]
    base_path = "/MoEITS/simplified_models/"
    output_path = "./models/prueba/"
    
    models = ["/MoEITS/simplified_models/qwen1.5-MoE-A2.7B-Chat-f1.25-mprod"]
    output_dirs = ["./models/prueba/qwen1.5-MoE-A2.7B-Chat-f1.25-mprod_retrained5"]

    model_name = os.path.join(base_path, sys.argv[1])
    exp = sys.argv[2]
    print("Training ", model_name)
    output_dir = os.path.join(output_path, model_name+'_retrained_'+exp)
    train(model_name, output_dir)
    

    # Cuando el pipeline funcione, cambia `train_dataset=large_dataset`
    # y ajusta epochs/lr según tus necesidades.
