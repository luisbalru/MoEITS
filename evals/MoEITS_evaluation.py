from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re
from typing import List

class MoEITSEvaluation(DeepEvalBaseLLM):
    def __init__(self, model_path="Qwen/Qwen1.5-MoE-A2.7B"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🌊 Loading Model: {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True # Opcional: seguridad extra
            )
        
        # CRITICAL FOR BATCHING: Set padding side to left for decoder-only models
        self.tokenizer.padding_side = "left" 
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
            attn_implementation="eager"        
        )
        self.model.eval()

    def load_model(self):
        return self.model

    def get_model_name(self):
        return self.model_path

    def _apply_chat_template(self, prompt: str) -> str:
        """Helper to format a single prompt."""
        # Detect context for specific system prompts
        is_math = "math" in prompt.lower() or "calculation" in prompt.lower()
        
        if is_math:
            system_msg = "You are a helpful assistant. Solve the math problem step by step."
        else:
            system_msg = "You are a helpful assistant. Answer the multiple choice question by outputting ONLY the correct option letter (A, B, C, or D). Do not explain."

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _clean_multiple_choice(self, text):
        """Extracts just 'A', 'B', 'C', 'D'."""
        text = text.strip()
        # Look for single letter answer or "Answer: A" pattern
        match = re.search(r'\b([A-D])\b', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return text

    def generate(self, prompt: str) -> str:
        """Fallback for single generation."""
        return self.batch_generate([prompt])[0]

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        The method DeepEval calls when batch_size > 1.
        """
        # 1. Apply Chat Template to all prompts
        chat_prompts = [self._apply_chat_template(p) for p in prompts]

        # 2. Tokenize with Padding (Crucial for batching)
        inputs = self.tokenizer(
            chat_prompts, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)

        # 3. Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=20, # Safe upper limit
                do_sample=False,    # Greedy
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=False
            )

        # 4. Decode and Strip Input Tokens
        results = []
        input_length = inputs.input_ids.shape[1]
        
        for i, output_ids in enumerate(generated_ids):
            # Slice off the input prompt
            new_tokens = output_ids[input_length:]
            decoded_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            # Apply cleaning only if it looks like a Multiple Choice Question
            # (Check original prompt for context)
            if "math" not in prompts[i].lower():
                decoded_text = self._clean_multiple_choice(decoded_text)
            
            results.append(decoded_text)

        return results

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    async def a_batch_generate(self, prompts: List[str]) -> List[str]:
        return self.batch_generate(prompts)

    def generate_samples(self, prompt: str, n: int, temperature: float) -> List[str]:
        """Specific for HumanEval (Pass@k)"""
        formatted_prompt = self._apply_chat_template(prompt)
        inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                num_return_sequences=n,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id
            )

        completions = []
        input_len = inputs.input_ids.shape[1]
        for i in range(generated_ids.shape[0]):
            output_ids = generated_ids[i][input_len:]
            completions.append(self.tokenizer.decode(output_ids, skip_special_tokens=True))
            
        return completions