import torch
from deepeval.models import DeepEvalBaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

class MoEITSEvaluation(DeepEvalBaseLLM):
    def __init__(self, model_path: str):
        print(f"Loading Pruned Model with Flash Attention 2 from: {model_path}")
        
        # 1. Load Tokenizer (Trust remote code for custom tokenizers)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load Model with Flash Attention 2
        # 'trust_remote_code=True' -> Executes your custom modeling_xxx.py
        # 'attn_implementation="flash_attention_2"' -> The speed secret sauce
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True, 
            #attn_implementation="flash_attention_2" 
        ).eval()

        self.model_name = model_path.split("/")[-1]

    def load_model(self):
        return self.model

    # --- BATCH GENERATION (Used by MMLU, HellaSwag, ARC, etc.) ---
    def batch_generate(self, prompts: List[str]) -> List[str]:
        # Batch tokenize
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512,  # Adjust based on benchmark needs
                do_sample=False,     # Deterministic for benchmarks
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode and strip the prompt from the output
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        final_outputs = []
        for prompt, prediction in zip(prompts, decoded_outputs):
            # Strict cleaning ensures we only grade the *new* text
            cleaned = prediction.replace(prompt, "").strip()
            final_outputs.append(cleaned)
            
        return final_outputs

    # --- SINGLE GENERATION (Used as fallback) ---
    def generate(self, prompt: str) -> str:
        return self.batch_generate([prompt])[0]
    
    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronous generation method required by DeepEvalBaseLLM.
        Since we are using local HF transformers (sync), we just wrap the sync call.
        """
        return self.generate(prompt)

    # --- SAMPLING GENERATION (Required ONLY for HumanEval) ---
    def generate_samples(self, prompt: str, n: int) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=True,      # Must sample for pass@k
                temperature=0.8,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_model_name(self):
        return self.model_name