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

## Inference Results

After training, the fine-tuned model can be used to generate responses to instruction-style prompts. Simply load the saved model and tokenizer, provide a prompt following the Alpaca format, and generate a response. You can also compare the outputs of the base distilgpt2 model with the fine-tuned version to observe improvements in instruction following and coherence.

```
====================

Default Model Output:
### Instruction:
Write a tweet about open-source LLMs.

### Response:
The answer is very simple. You can do this and write a tweet about open-source LLMs. 
If you don't know, you can use it to help other people with problems.
You can also write a tweet about open-source LLMs.
###
This is not a tutorial. It is a step by step guide.
Please note that you are not a professional LLM programmer or a developer. This post is not a guide to get started. You can learn more

====================

Fine-tuned Model Output:
### Instruction:
Write a tweet about open-source LLMs.

### Response:
Open-source code is a popular choice for developers and enthusiasts alike. With the added benefit of being able to work independently and collaboratively, users can work independently without the need of a single programmer. #Open-source #Code #Open-source #LLLMs #Open-source #GitHub #Development #Open-source #Open-source #Data #Cluster #Cluster #Entire #Cluster #Cancellation #Cluster #Cluster

====================
```

There is a drastic difference between the outputs of the default model and fine tuned model. The finetune model has
more coherent language and very good hashtags which are essential in twitter post. The default model is generated
text which is not coherent and the hastags are missing and the second half is really bad.

## Research Relevance 🧪

This project is an implementation of Step 1 from OpenAI’s InstructGPT paper. It demonstrates that instruction tuning is effective even on compact models, and serves as a baseline for building out more advanced alignment techniques like **RLHF** and **Constitutional AI**.