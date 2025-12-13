from transformers import AutoModelForCausalLM, AutoTokenizer
from moeits.models.qwen2_moe import Qwen2MoeForCausalLM
import sys

if __name__ == '__main__':
    tokenizer_path = sys.argv[1]
    model_path = sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    or_model = AutoModelForCausalLM.from_pretrained(tokenizer_path, device_map = "auto")

    simp_model = Qwen2MoeForCausalLM.from_pretrained(model_path, device_map = "auto")

    text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"

    print("LLM EVALUATION")
    print("Text: ", text)

    print("#################### Original Model")

    inputs = tokenizer(text, return_tensors="pt")
    outputs = or_model.generate(**inputs.to(or_model.device), max_new_tokens=100)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)

    print("#################### Simplified Model")

    inputs = tokenizer(text, return_tensors="pt")
    outputs = simp_model.generate(**inputs.to(simp_model.device), max_new_tokens=100)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)
