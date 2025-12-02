from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

class MoEITSEvaluation(DeepEvalBaseLLM):
    def __init__(self, model_path="Qwen/Qwen1.5-MoE-A2.7B", quantization=None):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model.eval()

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        """
        DeepEval passes a raw string 'prompt'. We must:
        1. Wrap it in Qwen's Chat Template.
        2. Force the model to be concise (System Prompt).
        3. Clean the output so 'The answer is B' becomes 'B'.
        """
        
        # 1. Define a "Constrained" System Prompt to force brevity
        # Use this for MMLU, ARC, HellaSwag (Multiple Choice)
        system_prompt = "You are a helpful assistant. You are answering multiple choice questions. Output ONLY the single letter corresponding to the correct answer (e.g. A, B, C, or D). Do not provide explanations."
        
        # NOTE: For GSM8K (Math), you might need a different system prompt 
        # (e.g. "Think step by step") and disable the cleaning logic below.
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # 2. Apply Chat Template (Crucial for Qwen)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=20, # Keep it short for MMLU/ARC
                do_sample=False,   # Greedy decoding is better for benchmarks
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 3. Decode and Extract
        # Remove the input tokens from the output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 4. Post-Processing / Cleaning
        cleaned_response = self._clean_response(response)
        
        # Debugging: Print to see what's happening if it fails again
        # print(f"DEBUG: Raw='{response.strip()}' -> Clean='{cleaned_response}'")
        
        return cleaned_response

    def _clean_response(self, text):
        """
        Extracts the first valid option letter (A, B, C, D, E) from the response.
        """
        text = text.strip()
        
        # If the model outputs "The answer is A", regex grabs "A"
        match = re.search(r'\b([A-E])\b', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
        # Fallback: return the first character if it's a letter
        if len(text) > 0 and text[0].isalpha():
            return text[0].upper()
            
        return text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_path