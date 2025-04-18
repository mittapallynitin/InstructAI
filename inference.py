import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./distilgpt2-alpaca/checkpoint-9000"

tokenizer = AutoTokenizer.from_pretrained(model_path)
fine_tuned_model = AutoModelForCausalLM.from_pretrained(model_path)
default_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

fine_tuned_model.eval()
default_model.eval()


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
fine_tuned_model.config.pad_token_id = tokenizer.pad_token_id
default_model.config.pad_token_id = tokenizer.pad_token_id

fine_tuned_model.to(device)
default_model.to(device)


def generate_response(prompt, model, tokenizer, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

instruction = "Write a tweet about open-source LLMs."

prompt = f"""### Instruction:
{instruction}

### Response:"""

output1 = generate_response(prompt, default_model, tokenizer)
output2 = generate_response(prompt, fine_tuned_model, tokenizer)
print("\nn" + "=" * 20 + "\n")
print("Default Model Output:")
print(output1)
print("\n" + "=" * 20 + "\n")
print("Fine-tuned Model Output:")
print(output2)
print("\n" + "=" * 20 + "\n")