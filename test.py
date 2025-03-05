import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
import random
from random import randrange

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,LlamaForCausalLM
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

import os
import numpy as np

def seed_torch(seed=3404):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())  # 参数总量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数量
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
checkpoint="../llama3/Llama-3.2-1B-Instruct"
# checkpoint="../llama3/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token = "<pad>"
# tokenizer.pad_token_id = 0
# tokenizer.add_special_tokens({"pad_token": "<pad>"})
# tokenizer.pad_token = "<pad>"
# special_tokens = tokenizer.special_tokens_map
# print("特殊token映射:", special_tokens)
print(tokenizer.pad_token,tokenizer.pad_token_id,tokenizer.decode([0]))
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(checkpoint)
print("Model vocab size:", model.config.vocab_size)
print("Tokenizer vocab size:", tokenizer.vocab_size)
print(model)
print_model_parameters(model)


def train():
    dataset = load_dataset("../llama3/data/databricks-dolly-15k", split="train")
    print(dataset)
    def preprocess_function(examples):
        sources = [
            f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:"
            if context
            else f"### Instruction:\n{instruction}\n\n### Response:"
            for instruction, context in zip(examples["instruction"], examples["context"])
        ]
        targets = examples["response"]
        return {"sources": sources, "targets": targets}
    # remove_columns = ["instruction", "context", "response","category"]
    # dataset = dataset.map(preprocess_function, batched=True,remove_columns=remove_columns)
    # print(dataset[0])
    def encode_function(examples):
        sources = [
            f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:"
            if context
            else f"### Instruction:\n{instruction}\n\n### Response:"
            for instruction, context in zip(examples["instruction"], examples["context"])
        ]
        targets = [f'{response}'for response in examples["response"]]
        model_inputs = tokenizer(
            sources,
            max_length=128,#512
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        labels = tokenizer(
            targets,
            max_length=128,#512
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_columns = ["instruction", "context", "response", "category"]
    dataset = dataset.map(encode_function, batched=True,remove_columns=remove_columns)
    dataset.set_format(type='torch')
    print(dataset[0].keys(),type(dataset[0]))

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)#bs_max=10
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            print(input_ids)
            print("input_ids、attention_mask、labels：",input_ids.shape,attention_mask.shape,labels.shape)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            print(outputs.keys())#loss,logits,past_key_values
            print("logits:",outputs.logits.shape)
            # print("past_key_values:",outputs.past_key_values)

            ###compute loss
            ignore_index=-100
            logits = outputs.logits.float()
            # print(logits)
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            # print(labels)
            shift_labels = labels[..., 1:].contiguous()
            logits_reshaped = logits.view(-1, logits.size(-1))
            labels_reshaped = shift_labels.view(-1)
            # loss_fn = nn.CrossEntropyLoss()

            loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
            # loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fn(logits_reshaped, labels_reshaped)
            print("compute loss:",loss)

            loss = outputs.loss
            print("model loss",loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            break
        break
        print(f"Epoch {epoch + 1} - Average Loss: {total_loss / len(train_dataloader)}")


def SFT_train():
    dataset = load_dataset("../llama3/data/databricks-dolly-15k", split="train")
    print(dataset)
    print(f"dataset size: {len(dataset)}")
    print(dataset[randrange(len(dataset))])

    def format_instruction(sample):
        return f"""### Instruction:
    Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

    ### Input:
    {sample['response']}

    ### Response:
    {sample['instruction']}
    """

    print(format_instruction(dataset[randrange(len(dataset))]))

def test1():
    ignore_index = -100
    labels = torch.tensor([0, 1, 1, 1])
    print(labels)
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    print(labels)
    shift_labels = labels[..., 1:].contiguous()
    print(shift_labels)
if __name__ == '__main__':
    seed_torch(3404)
    train()
    # test1()
    # SFT_train()