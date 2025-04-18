# Import necessary libraries
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

import datasetup

checkpoint = "distilgpt2"
device = "mps"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# to avoid padding warning
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

formatter = datasetup.AlpacaFormater(tokenizer)

alpaca_dataset = datasetup.get_alpaca_dataset()
alpaca_tokenized = alpaca_dataset.map(
    formatter,
    batched=False,
    remove_columns=alpaca_dataset.column_names,
)

training_args = TrainingArguments(
    output_dir="./distilgpt2-alpaca",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="no",
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=False,
    report_to="none",
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=alpaca_tokenized,
    tokenizer=tokenizer,
    data_collator=formatter.collator(),
)
# 6. Train the model
trainer.train()

