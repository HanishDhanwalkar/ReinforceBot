from trl import PPOConfig

config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
)

from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# pretokenize our dataset using the tokenizer
def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample

dataset = dataset.map(tokenize, batched=False)