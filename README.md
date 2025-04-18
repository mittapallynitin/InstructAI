# Instruction tuning - OpenAI Research Paper 

```
Research Paper: Training language models to follow instructions with human feedback
Publisher: OpenAI (2022)
```

## 🚀 Project Highlights ( TLDR; )

- 🔧 Full fine-tuning of DistilGPT2 using Hugging Face `Trainer`
- 🦙 Instruction tuning with the Alpaca dataset
- 🧠 Inspired by the InstructAI from OpenAI
- ⚙️ Ready for training on GPU or Apple M1/M2 with MPS support
- 📦 Clean modular code with dataset handling, training, and inference

## Project Structure
```
InstructAI                                            
├─  datasetup.py  
├─  inference.py  
├─  model.py                                        
├─  train.py
├─  distilgpt2-alpaca
│     └─ model savepoints...
```

## Introduction

In recent years, large language models (LLMs) have achieved remarkable capabilities, but they often fail to follow user instructions reliably. OpenAI’s InstructGPT paper introduced a powerful solution: instruction tuning with human feedback. In this project, I will implement the first stage of that process — instruction fine-tuning — on a compact model: DistilGPT2

## Paper Summary: InstructGPT 📃

**Citation**: [Training language models to follow instructions with human feedback, OpenAI (2022)](https://arxiv.org/abs/2203.02155)

**Key Idea**: Instead of relying on generic language modeling, train models on task-specific, instruction-response datasets. InstructGPT introduced a 3-step pipeline

- **SFT**: Fine-tune the base model on instruction-following examples.
- **Reward Model**: Train a model to score responses based on human preferences.
- **RLHF**: Fine-tune the model using reinforcement learning with the reward model as the reward function.


## What Is Instruction Tuning? ✽

Instruction tuning teaches a language model to better understand and execute user instructions. Instead of continuing arbitrary text, the model learns to respond to prompts and maintain conversation instead of statistically completing the prompt. 


## Model Choice: Why DistilGPT2? 𝌭

While the original paper used GPT-3 (175B parameters), I chose DistilGPT2 (82M) to:
- Reduce compute requirements
- Quickly iterate on experiments
- Show instruction tuning works even on small models

**Resources Available**: Apple M2 with 16GB RAM. 


## Dataset: Stanford Alpaca (GPT-Generated Instructions) 📉

The Alpaca dataset by Stanford contains ~52,000 instruction-response pairs generated using GPT-3. It’s structured and clean, making it perfect for training smaller models.

Each sample has:
- **Instruction**: The task to be performed
- **Input**: Additional context (optional)
- **Output**: Expected answer

**Training example**

```
### Instruction:
Re-arrange the sentence to create variety.

### Input:
The setting sun bathed the beach in a golden shimmer.

### Response:
The beach glowed golden in the setting sun.
```

## Training Setup &#129303; 🔥

**Stack**

```
    • Huggingface 🤗 Datasets
    • Huggingface 🤗 Transformers & Trainer
    • PyTorch
    • DistilGPT2
    • Alpaca dataset
```

**Training Objective:**
Minimize cross-entropy loss over input sequences. 

## Research Relevance 🧪

This project is an implementation of Step 1 from OpenAI’s InstructGPT paper. It demonstrates that instruction tuning is effective even on compact models, and serves as a baseline for building out more advanced alignment techniques like **RLHF** and **Constitutional AI**.