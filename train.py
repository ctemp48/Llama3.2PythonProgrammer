from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import torch

# Load dataset
dataset_name = "Vezora/Tested-143k-Python-Alpaca"
dataset = load_dataset(dataset_name, split="train").select(range(100000))

# Preprocess dataset
def format_example(example):
    return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"}

dataset = dataset.map(format_example, remove_columns=["instruction", "input", "output"])

# Load model and tokenizer with 4-bit quantization

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

# Configure LoRa
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=2e-4,
    bf16=True,
    optim="adamw_bnb_8bit",
    report_to="none"
)

model.config.use_cache = False

# Trainer setup
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args
)

# Start training
trainer.train()

# Save model
trainer.model.save_pretrained("./fine-tuned-llama-3-1b")
tokenizer.save_pretrained("./fine-tuned-llama-3-1b")

