import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# -----------------------------
# 1️⃣ Configuración
# -----------------------------
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
OUTPUT_PATH = "./models/prueba/mixtral_retrained"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# -----------------------------
# 2️⃣ Tokenizer y modelo
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", dtype="bfloat16")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# -----------------------------
# 3️⃣ Congelar todo excepto routers y expertos
# -----------------------------
for p in model.parameters():
    p.requires_grad = False  # congelar todo

# Función para habilitar routers y al menos 1 experto por capa
def enable_gate_and_experts(model):
    found = False
    for module in model.modules():
        # Routers
        if hasattr(module, "gate"):
            for p in module.router.parameters():
                p.requires_grad = True
            found = True
        # Experts (solo el primero de cada capa)
        if hasattr(module, "experts") and len(module.experts) > 0:
            for p in module.experts[0].parameters():
                p.requires_grad = True
            found = True
    if not found:
        raise RuntimeError("No routers or experts found in the model.")
    print("✅ Routers and first expert per layer are trainable.")

enable_gate_and_experts(model)

# -----------------------------
# 4 Dataset sintético
# -----------------------------
texts = ["Testing Mixtral8X7B pipeline with DeepSpeed."] * 500  # mínimo seguro para debug

dataset = Dataset.from_dict({"text": texts})

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=32  # ⚡ short for testing, avoids OOM and shape issues
    )

dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"]
)

# -----------------------------
# 4️⃣ Collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8  # important for bfloat16 & DeepSpeed
)

# -----------------------------
# 5️⃣ Custom Trainer
# -----------------------------
class MoeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        # Handle optional MoE aux losses safely
        aux_loss = getattr(outputs, "router_aux_loss", None) or getattr(outputs, "aux_loss", None)
        if aux_loss is not None and aux_loss.numel() > 0:
            loss = loss + 0.01 * aux_loss

        return (loss, outputs) if return_outputs else loss

# -----------------------------
# 6️⃣ Sanity check trainable params
# -----------------------------
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable}/{total}")
if trainable == 0:
    raise RuntimeError("No trainable parameters connected to the loss. Check model requires_grad flags.")

# -----------------------------
# 7️⃣ TrainingArguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=1000,
    save_total_limit=2,
    bf16=True,
    optim="adamw_torch",
    max_grad_norm=1.0,
    report_to="none",
    deepspeed="./ds_config.json",  # ⚡ make sure ds_config.json exists
)

# -----------------------------
# 8️⃣ Trainer initialization
# -----------------------------
trainer = MoeTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# -----------------------------
# 9️⃣ Start training
# -----------------------------
print("🚀 Starting training...")
trainer.train()
print("✅ Training completed successfully!")
