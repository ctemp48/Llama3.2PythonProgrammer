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
seed = 42
set_seed(seed)

load_dotenv()
wandb.login(key=os.getenv('wandb-key'))

CUTOFF_LEN = 256

dataset = load_dataset("gbharti/wealth-alpaca_lora")
dataset = dataset['train'].train_test_split(test_size = 0.2)

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
    )

model_name="meta-llama/Llama-3.2-3B-Instruct"
device_map = {"": 0}
original_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                      device_map = device_map,
                                                      quantization_config=bnb_config,
                                                      use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
tokenizer.pad_token = 0

def generate_and_tokenize_prompt(data_point):
    """This function masks out the labels for the input, so that our loss is computed only on the
    response."""
    if data_point['input']:
        user_prompt = 'Below is an instruction that describes a task, paired with an input that ' \
                      'provides further context. Write a response that appropriately completes ' \
                      'the request.\n\n'
        user_prompt += f'### Instruction:\n{data_point["instruction"]}\n\n'
        user_prompt += f'### Input:\n{data_point["input"]}\n\n'
        user_prompt += f'### Response:\n'
    else:
        user_prompt = 'Below is an instruction that describes a task. Write a response that ' \
                      'appropriately completes the request.'
        user_prompt += f'### Instruction:\n{data_point["instruction"]}\n\n'
        user_prompt += f'### Response:\n'

    # Count the length of prompt tokens
    len_user_prompt_tokens = len(tokenizer(user_prompt,
                                           truncation=True,
                                           max_length=CUTOFF_LEN + 1,
                                           padding='max_length')['input_ids'])
    len_user_prompt_tokens -= 1  # Minus 1 (one) for eos token

    # Tokenise the input, both prompt and output
    full_tokens = tokenizer(
        user_prompt + data_point['output'],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding='max_length',
    )['input_ids'][:-1]
    return {
        'input_ids': full_tokens,
        'labels': [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
        'attention_mask': [1] * (len(full_tokens)),
    }


train_dataset = dataset['train'].map(generate_and_tokenize_prompt)
test_dataset = dataset['test'].map(generate_and_tokenize_prompt)

original_model = prepare_model_for_kbit_training(original_model)

config = LoraConfig(  #fiddle around with these
    r=8, #Rank
    lora_alpha=8,
    target_modules="all-linear",
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(original_model, config)

#print(peft_model.print_trainable_parameters())

output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments( #fiddle around with these
    seed=seed,
    data_seed=seed,
    output_dir = output_dir,
    warmup_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_steps=20,
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
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
wandb.finish()

