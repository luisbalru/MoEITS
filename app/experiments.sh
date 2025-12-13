#!/bin/bash

# Mixtral 8x7B Instruct
python3 main.py mixtral 7.5 prod
python3 main.py mixtral 5 prod
python3 main.py mixtral 2.5 prod
python3 main.py mixtral 1.25 prod



# Qwen1.5-MoE-A2.7B-Chat
python3 main.py qwen1.5 7.5 prod
python3 main.py qwen1.5 5 prod
python3 main.py qwen1.5 2.5 prod
python3 main.py qwen1.5 1.25 prod


# DeepSeek-V2-Lite-Chat
python3 main.py deepseek 7.5 prod
python3 main.py deepseek 5 prod
python3 main.py deepseek 2.5 prod
python3 main.py deepseek 1.25 prod