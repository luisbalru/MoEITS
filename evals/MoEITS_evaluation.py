from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoTokenizer
from moeits.models.qwen2_moe import Qwen2MoeForCausalLM
import torch
import re
from typing import List

class MoEITSEvaluation(DeepEvalBaseLLM):
    def __init__(self, model_path, tokenizer_path="Qwen/Qwen1.5-MoE-A2.7B"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🌊 Loading Model: {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self.tokenizer.padding_side = "left" 
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Carga del modelo (sin cambios)
        if 'qwen1.5' in model_path.lower() or 'qwen2' in model_path.lower():
            self.model = Qwen2MoeForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                dtype=torch.float16, 
                trust_remote_code=True,
                attn_implementation="eager"        
            )
        else:
            # Fallback para otros modelos si fuera necesario
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
            )
        self.model.eval()

    def load_model(self):
        return self.model

    def get_model_name(self):
        return self.model_path

    def _determine_prompt_type(self, prompt: str):
        """Heurística simple para determinar cómo procesar el prompt."""
        p_lower = prompt.lower()
        if "math" in p_lower or "calculation" in p_lower or "gsm8k" in p_lower:
            return "reasoning"
        if "true" in p_lower and "false" in p_lower: # BoolQ pattern
            return "boolean"
        return "mcq" # Default to Multiple Choice

    def _apply_chat_template(self, prompt: str) -> str:
        """
        Usa un System Prompt más permisivo para permitir CoT.
        La limpieza se hace AFTER generation.
        """
        # System prompt genérico que funciona bien para Qwen
        system_msg = "You are a helpful assistant."
        
        # Si detectamos que es explícitamente razonamiento matemático
        if "math" in prompt.lower():
             system_msg += " Think step by step before giving the final answer."

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _process_output(self, output_text, prompt_type):
        """Limpieza robusta dependiendo del tipo de tarea."""
        text = output_text.strip()

        # 1. Lógica para Multiple Choice (MMLU, ARC, HellaSwag)
        if prompt_type == "mcq":
            # Busca patrones como "Answer: A", "option B", o simplemente "C" al final.
            # Prioriza encontrar la respuesta explícita.
            match = re.search(r'Answer:\s*([A-D])', text, re.IGNORECASE)
            if match: return match.group(1).upper()
            
            match = re.search(r'\b([A-D])\.', text) # Ex: "A."
            if match: return match.group(1).upper()
            
            # Último recurso: busca una letra solitaria
            match = re.search(r'\b([A-D])\b', text, re.IGNORECASE)
            if match: return match.group(1).upper()
            return text

        # 2. Lógica para Boolean (BoolQ)
        elif prompt_type == "boolean":
            text_lower = text.lower()
            if "true" in text_lower or "yes" in text_lower: return "true"
            if "false" in text_lower or "no" in text_lower: return "false"
            return text

        # 3. Lógica para Razonamiento (GSM8K, MathQA)
        # DeepEval a veces busca el número exacto, dejemos el texto completo 
        # o intentemos extraer el último número si es muy verboso.
        return text 

    def generate(self, prompt: str) -> str:
        return self.batch_generate([prompt])[0]

    def batch_generate(self, prompts: List[str]) -> List[str]:
        # 1. Detectar tipo de tarea basado en el primer prompt (asumimos batch homogéneo)
        prompt_type = self._determine_prompt_type(prompts[0])
        
        # 2. Configurar tokens dinámicamente
        # Si es razonamiento, necesitamos MUCHO espacio. Si es MCQ, menos.
        max_tokens = 1024 if prompt_type == "reasoning" else 100

        chat_prompts = [self._apply_chat_template(p) for p in prompts]

        inputs = self.tokenizer(
            chat_prompts, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,  # AUMENTADO DE 20 A 1024/100
                do_sample=False,            # Greedy sigue siendo mejor para benchmarks
                temperature=0.0,            # Asegurar determinismo
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True              # Más rápido
            )

        results = []
        input_length = inputs.input_ids.shape[1]
        
        for i, output_ids in enumerate(generated_ids):
            new_tokens = output_ids[input_length:]
            decoded_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Aplicar limpieza específica
            if "lambada" in prompts[i].lower():
                 # LAMBADA requiere la palabra exacta, no tocar mucho
                 final_output = decoded_text.strip()
            else:
                 final_output = self._process_output(decoded_text, prompt_type)
            
            results.append(final_output)

        return results

    # ... (Resto de métodos async y generate_samples igual) ...
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    async def a_batch_generate(self, prompts: List[str]) -> List[str]:
        return self.batch_generate(prompts)

    def generate_samples(self, prompt: str, n: int, temperature: float) -> List[str]:
        # Para HumanEval no cambiamos mucho, pero aseguramos max_tokens alto
        formatted_prompt = self._apply_chat_template(prompt)
        inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024, # Importante para código
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