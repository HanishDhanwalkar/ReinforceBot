import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

def chat():
    # Load the trained model
    model = GPT2LMHeadModel.from_pretrained("my_grpo_model/checkpoint-3")
    tokenizer = GPT2Tokenizer.from_pretrained("my_grpo_model/checkpoint-3")
    
    print("Chat with RL-trained bot! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
        # Generate response
        inputs = tokenizer(user_input, return_tensors="pt")
        response_ids = model.generate(**inputs, max_length=128)
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        print(f"\nBot: {response}")

# Start chatting
chat()