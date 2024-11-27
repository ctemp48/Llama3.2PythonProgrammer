from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig,
    DataCollatorForLanguageModeling,
    set_seed
)
from tqdm import tqdm
from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np
from huggingface_hub import interpreter_login
import os
from dotenv import load_dotenv
from functools import partial
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

#interpreter_login()

load_dotenv()
wandb.login(key=os.getenv('wandb-key'))

dataset = load_dataset("gbharti/wealth-alpaca_lora")
dataset = dataset['train'].train_test_split(test_size = 0.2)

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

model_name="meta-llama/Llama-3.2-3B-Instruct"
device_map = {"": 0}
original_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                      device_map = device_map,
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,padding_side="left",add_eos_token=True,add_bos_token=True,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def create_prompt_formats(sample):
    if sample['input']:
        text = 'Below is an instruction that describes a task, paired with an input that provides' \
               ' further context. Write a response that appropriately completes the request.\n\n'
        text += f'### Instruction:\n{sample["instruction"]}\n\n'
        text += f'### Input:\n{sample["input"]}\n\n'
        text += f'### Response:\n{sample["output"]}'
        sample['text'] = text
        return sample
    
    else:
        text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
        text += f'### Instruction:\n{sample["instruction"]}\n\n'
        text += f'### Response:\n{sample["output"]}'
        sample['text'] = text
        return sample


def get_max_length(model):
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int,seed, dataset):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)#, batched=True)
    
    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=['instruction', 'output', 'input'],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

seed = 42
set_seed(seed)

max_length = get_max_length(original_model)

train_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset['train'])
test_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset['test'])

original_model = prepare_model_for_kbit_training(original_model)

config = LoraConfig(  #fiddle around with these
    r=32, #Rank
    lora_alpha=32,
    target_modules="all-linear",
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

original_model.gradient_checkpointing_enable()

peft_model = get_peft_model(original_model, config)

#print(peft_model.print_trainable_parameters())

output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments( #fiddle around with these
    seed=seed,
    data_seed=seed,
    output_dir = output_dir,
    warmup_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=1000,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps=25,
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=25,
    evaluation_strategy="steps",
    eval_steps=25,
    do_eval=True,
    gradient_checkpointing=True,
    report_to="none",
    overwrite_output_dir = 'True',
    group_by_length=True,
)

peft_model.config.use_cache = False

peft_trainer = Trainer(
    model=peft_model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=peft_training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

peft_trainer.train()

