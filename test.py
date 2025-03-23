import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset,Dataset
import random
from random import randrange

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,LlamaForCausalLM,BertForSequenceClassification,T5ForConditionalGeneration
from transformers import TrainingArguments,DataCollatorForLanguageModeling
from trl import SFTTrainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

import os
import numpy as np
import pandas as pd
import copy

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
use_flash_attention  = False
model = AutoModelForCausalLM.from_pretrained(checkpoint,use_cache=False,use_flash_attention_2=use_flash_attention,torch_dtype=torch.float16,device_map="auto")
model.config.pretraining_tp = 1
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
    idx = list(range(len(dataset)))
    dataset = Dataset.from_dict({'input_ids': dataset['input_ids'],
                                       'labels': dataset['labels'],
                                       'attention_mask': dataset['attention_mask'],
                                       'idx': idx})
    dataset.set_format(type='torch')
    print(dataset[0].keys(),type(dataset[0]))

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)#bs_max=10
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
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
            print("compute loss:",loss.item())

            loss = outputs.loss
            print("model loss",loss.item())

            loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            break
        break
        print(f"Epoch {epoch + 1} - Average Loss: {total_loss / len(train_dataloader)}")


def SFT_train():
    dataset = load_dataset("../llama3/data/databricks-dolly-15k", split="train")
    print(dataset)
    train_dataset =  dataset.shuffle(seed=3404).select(range(1000))
    print(train_dataset)


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
        targets = [f'{response}' for response in examples["response"]]
        model_inputs = tokenizer(
            sources,
            max_length=128,  # 512
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        labels = tokenizer(
            targets,
            max_length=128,  # 512
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # remove_columns = ["instruction", "context", "response", "category"]
    # dataset = dataset.map(encode_function, batched=True, remove_columns=remove_columns)
    # dataset.set_format(type='torch')
    # print(dataset[0].keys(), type(dataset[0]))
    args = TrainingArguments(
        output_dir="llama-7-int4-dolly",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=1,
        save_strategy="no",
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=True
    )
    max_seq_length = 2048  # max sequence length for model and packing of the dataset

    def format_instruction(sample):
        return f"""### Instruction:
    Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

    ### Input:
    {sample['response']}

    ### Response:
    {sample['instruction']}
    """
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        # peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=format_instruction,
        args=args,
    )
    trainer.train()

def CausalLM_train():
    dataset = load_dataset("../llama3/data/databricks-dolly-15k", split="train")
    dataset = dataset.shuffle(seed=3404).select(range(100))
    print(dataset)
    def tokenize_function(examples):
        # sample = f"### Instruction:\n{examples['instruction']}\n\n### Context:\n{examples['context']}\n\n### Response:\n{examples['response']}" if examples['context'] else f"### Instruction:\n{examples['instruction']}\n\n### Response:\n{examples['response']}"
        # return tokenizer(examples['instruction'], truncation=True, max_length=512, padding="max_length")

        sources = [
            f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{examples['response']}"
            if context
            else f"### Instruction:\n{instruction}\n\n### Response:\n{examples['response']}"
            for instruction, context in zip(examples["instruction"], examples["context"])
        ]
        model_inputs = tokenizer(
            sources,
            max_length=512,  # 512
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return model_inputs

    remove_columns = ["instruction", "context", "response", "category"]
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=remove_columns)
    dataset.set_format(type='torch')
    print(dataset[0].keys(), type(dataset[0]))
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 设置为 False，表示使用因果语言模型
    )
    sample = dataset[0]

    # 使用 data_collator 整理数据
    batch = data_collator([sample])

    # 打印 input_ids 和 labels
    print("Input IDs:", batch["input_ids"])
    print("Labels:", batch["labels"])
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model.to(device)

    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            print(batch)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            print(loss)
            progress_bar.set_postfix({"loss": loss.item()})
            break
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
def test1():
    ignore_index = -100
    labels = torch.tensor([0, 1, 1, 1])
    print(labels)
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    print(labels)
    shift_labels = labels[..., 1:].contiguous()
    print(shift_labels)

def CausalLM_lp():
    dataset = load_dataset("../llama3/data/databricks-dolly-15k", split="train")
    print(dataset)
    dataset = dataset.shuffle(seed=3404).select(range(100))
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
    idx = list(range(len(dataset)))
    dataset = Dataset.from_dict({'input_ids': dataset['input_ids'],
                                       'labels': dataset['labels'],
                                       'attention_mask': dataset['attention_mask'],
                                       'idx': idx})
    dataset.set_format(type='torch')
    print(dataset[0].keys(),type(dataset[0]))

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)#bs_max=10
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_gap = [[] for _ in range(len(train_dataloader))]
    compress = 5e-8
    def loss_particles(model_lp,comp,loss_gap):
        compress = comp
        loss_g_before = {}
        iterator = iter(train_dataloader)
        trange = range(len(train_dataloader))
        before = tqdm(total=len(train_dataloader), desc=f"lp before{comp}")
        for step in trange:
            before.update(1)
            inputs = next(iterator)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = inputs["labels"].to(device)
            model_lp.eval()
            outputs = model_lp(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            step_idx = inputs["idx"]
            loss = outputs.loss
            for i in range(len(step_idx)):
                loss_g_before[step_idx[i].item()] = loss.item()
        before.close()

        with torch.no_grad():
            for name, module in model_lp.named_modules():
                if isinstance(module, (torch.nn.Linear)):
                    r = 1 - compress
                    module.weight.data = r * module.weight.data

        loss_g_after = {}
        iterator = iter(train_dataloader)
        trange = range(len(train_dataloader))
        after = tqdm(total=len(train_dataloader), desc=f"lp before{comp}")
        for step in trange:
            after.update(1)
            inputs = next(iterator)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = inputs["labels"].to(device)
            model_lp.eval()
            outputs = model_lp(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            step_idx = inputs["idx"]
            loss = outputs.loss
            for i in range(len(step_idx)):
                loss_g_after[step_idx[i].item()] = loss.item()
        after.close()

        keys = sorted(loss_g_before.keys())
        loss_g_gap = {key: loss_g_after[key]-loss_g_before[key] for key in keys}
        for key in keys:
            loss_gap[key].append(loss_g_gap[key])
        del model_lp
        # print(loss_gap)
    # model.to(device)
    for epoch in range(num_epochs):
        k = [2.5e-4]
        for k_i in k:
            model_lp = AutoModelForCausalLM.from_pretrained(checkpoint,use_cache=False,use_flash_attention_2=use_flash_attention,torch_dtype=torch.float16,device_map="auto")
            model_lp.config.pretraining_tp = 1
            model_lp.load_state_dict(copy.deepcopy(model.state_dict()))
            model_lp.to(next(model.parameters()).device)
            loss_particles(model_lp, k_i, loss_gap)
            del model_lp
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
            print("compute loss:",loss.item())

            loss = outputs.loss
            print("model loss",loss.item())

            loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            break
        break
        print(f"Epoch {epoch + 1} - Average Loss: {total_loss / len(train_dataloader)}")
    count_lp = len(loss_gap[0])
    print(count_lp)
    loss_g_file = f"lp.csv"
    df = pd.DataFrame(loss_gap, columns=[i for i in range(count_lp)])
    df.to_csv(loss_g_file, index=False)
if __name__ == '__main__':
    seed_torch(3404)
    # train()
    # test1()
    # SFT_train()
    # CausalLM_train()
    CausalLM_lp()