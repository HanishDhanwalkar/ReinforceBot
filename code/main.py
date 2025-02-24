import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, AutoModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Initialize the base model and tokenizer
# model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Initialize the PPO trainer
ppo_config = PPOConfig(
    # batch_size=1,  # Update after each interaction
    learning_rate=1e-5,
    # steps=1000,  # Total interactions
)
# ppo_trainer = PPOTrainer(model=model, args=ppo_config, tokenizer=tokenizer)
ppo_trainer = PPOTrainer(model=model, args=ppo_config)

def interactive_training():
    print("Start chatting! Type 'exit' to quit.")
    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
        # Generate a response
        inputs = tokenizer(user_input, return_tensors="pt")
        response_ids = model.generate(**inputs, max_length=128)
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        # Show response and get feedback
        print(f"\nBot: {response}")
        try:
            reward = float(input("Rate the response (e.g., 1.0 for good, -1.0 for bad): "))
        except:
            print("Invalid input. Using default reward 0.0.")
            reward = 0.0
        
        # Apply PPO update with the user's reward
        ppo_trainer.step([user_input], [response], [reward])
    
    # Save the model
    model.save_pretrained("custom_rl_gpt2")
    print("Model saved!")

# Start training
interactive_training()

