"""
Code Generation using CodeLLaMA via Ollama
pip install ollama
Requires: ollama pull codellama
"""
import ollama

def generate_code(prompt):
    res = ollama.chat(model="codellama", messages=[
        {"role": "system", "content": "Return only Python code, no explanation."},
        {"role": "user",   "content": prompt},
    ])
    return res["message"]["content"]

prompts = [
    "Write a Python function to compute fibonacci(n).",
    "Write a Python class Stack with push, pop, peek.",
    "Write a Python function to find the second largest number in a list.",
]

for p in prompts:
    print(f"PROMPT: {p}")
    print(f"CODE:\n{generate_code(p)}\n{'—'*50}")
