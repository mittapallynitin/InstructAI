import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "distilgpt2"
device = "mps"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# to avoid padding warning
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Set the model to evaluation mode
def generate_response(prompt):
    # Encode the input prompt
    model.eval()
    with torch.inference_mode():
        inputs = tokenizer(
            prompt, 
            padding=True,   
            truncation=True, 
            max_length=512,
            return_tensors="pt",
        ).to(device)

        # Generate a response from the model
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=100, do_sample=True,
            temperature=0.9)

        # Decode the generated response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response

# Example usage
if __name__ == "__main__":
    prompt = "how to kill a person"
    response = generate_response(prompt)
    print(response)