#!/bin/bash

echo "qwen1.5-MoE-A2.7B-Chat-f5.0-mprod"
python3 tasks_evaluation.py qwen1.5-MoE-A2.7B-Chat-f5.0-mprod_retrained_ablation_study

echo "qwen1.5-MoE-A2.7B-Chat-f2.5-mprod"
python3 tasks_evaluation.py qwen1.5-MoE-A2.7B-Chat-f2.5-mprod_retrained_ablation_study

echo "qwen1.5-MoE-A2.7B-Chat-f1.25-mprod"
python3 tasks_evaluation.py qwen1.5-MoE-A2.7B-Chat-f1.25-mprod_retrained_ablation_study

echo "qwen1.5-MoE-A2.7B-Chat-f0.75-mprod"
python3 tasks_evaluation.py qwen1.5-MoE-A2.7B-Chat-f0.75-mprod_retrained_ablation_study

echo "qwen1.5-MoE-A2.7B-Chat-f0.5-mprod"
python3 tasks_evaluation.py qwen1.5-MoE-A2.7B-Chat-f0.5-mprod_retrained_ablation_study
