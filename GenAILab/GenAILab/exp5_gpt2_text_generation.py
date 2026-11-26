"""
Text Generation with GPT-2 + Prompt Engineering
pip install transformers torch
"""
from transformers import pipeline

gen = pipeline("text-generation", model="gpt2", max_new_tokens=60)

prompts = {
    "Zero-shot"  : "Translate to French: Hello, how are you?",
    "Few-shot"   : "Q: Capital of France? A: Paris.\nQ: Capital of Japan? A:",
    "Chain-of-thought": "Think step by step. I have 10 apples, give 3 away. How many left?\nStep 1:",
    "Creative"   : "Once upon a time in a futuristic city,",
}

for style, prompt in prompts.items():
    out = gen(prompt, do_sample=True, temperature=0.7)[0]["generated_text"]
    print(f"[{style}]\n{out}\n{'-'*50}")
