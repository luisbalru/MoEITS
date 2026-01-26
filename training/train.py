import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)



class MoeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss

        # Compatibilidad Mixtral / Qwen / DeepSeek
        if hasattr(outputs, "router_aux_loss"):
            loss = loss + 0.01 * outputs.router_aux_loss
        elif hasattr(outputs, "aux_loss"):
            loss = loss + 0.01 * outputs.aux_loss

        return (loss, outputs) if return_outputs else loss

def freeze_all_but_router(model):
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "router" in name:
                param.requires_grad = True

def unfreeze_router_and_experts(model):
    for name, param in model.named_parameters():
        if "router" in name or "experts" in name:
            param.requires_grad = True

if __name__ == '__main__':
    MODEL_PATH = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    OUTPUT_PATH = "./models/prueba/mixtral_retrained"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    freeze_all_but_router(model)


    """
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train[:1]"
    )
    """
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train[:10]"
    )

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=2048
        )

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,              # FASE A
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=1,
        logging_steps=20,
        save_steps=1000,
        save_total_limit=2,
        bf16=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        report_to="none",
        deepspeed="./ds_config.json"
    )

    # TRAINING ROUTERS
    trainer = MoeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    print("TRAINING ROUTERS")
    trainer.train()

    # ROUTER + EXPERTS
    unfreeze_router_and_experts(model)
    trainer.args.learning_rate = 2e-5
    print("TRAINING ROUTERS AND EXPERTS")
    trainer.train()

    # GLOBAL FINE TUNING
    for param in model.parameters():
        param.requires_grad = True
    print("GLOBAL FINETUNING")
    trainer.args.learning_rate = 1e-5
    trainer.train()


    trainer.save_model(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
