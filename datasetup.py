from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer


def get_alpaca_dataset():
    """
    Load the Alpaca dataset from Hugging Face.
    """
    dataset = load_dataset("tatsu-lab/alpaca")["train"]
    return dataset

class AlpacaDataset(Dataset):
    def __init__(self):
        self.dataset = get_alpaca_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AlpacaDataLoader(DataLoader):
    def __init__(
        self, 
        dataset: AlpacaDataset, 
        tokenizer: PreTrainedTokenizer, 
        max_length: int = 512, 
        batch_size: int=1,
        shuffle: bool=True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        

    def collate_fn(self, batch: list[dict[str, str]]):
        
        input_texts = []
        for example in batch:
            instruction = example["instruction"]
            input = example.get("input", "").strip()
            output = example["output"]

            if input:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            input_texts.append(prompt)
        
        tokenized = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # GPT-style models: labels = input_ids (same)
        tokenized["labels"] = tokenized["input_ids"].clone()
        tokenized["text"] = input_texts

        return tokenized


class AlpacaFormater:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, example):
        return self.format_dataset(example)
    
    def format_dataset(self, example):
        instruction = example["instruction"]
        input_data = example.get("input", "").strip()
        output = example["output"].strip()

        if input_data:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_data}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        return self.tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    
    def collator(self):
        return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

if __name__ == "__main__":
    dataset = AlpacaDataset()
    print("Dataset loaded")

    # Load the tokenizer
    from transformers import AutoTokenizer
    
    checkpoint = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # to avoid padding warning
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")
    
    dataloader = AlpacaDataLoader(dataset, tokenizer)
    print("DataLoader created")
    
    for batch in dataloader:
        print(batch)
        break