import torch
# from transformers import GPT2LMHeadModel, , pipeline, AutoModel
# from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM
from transformers import GPT2Tokenizer


model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token


def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


    
if __name__ == "__main__":
    while True:
        input_text = input("Enter prompt: ")
        
        if input_text.lower() == "exit":
            break
        
        generated_text = generate_text(input_text)
        print(generated_text)