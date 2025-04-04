from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
from codebleu import calc_codebleu

# Paths
original_model_name = "meta-llama/Llama-3.2-1B-Instruct"
fine_tuned_model_path = "./fine-tuned-llama-3-1b"

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load original model
original_model = AutoModelForCausalLM.from_pretrained(original_model_name, quantization_config=bnb_config, device_map="auto")
original_tokenizer = AutoTokenizer.from_pretrained(original_model_name, padding_side="left")
original_tokenizer.pad_token = original_tokenizer.eos_token

# Load fine-tuned model
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path, quantization_config=bnb_config, device_map="auto")
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path, padding_side="left")
fine_tuned_tokenizer.pad_token = fine_tuned_tokenizer.eos_token

# Load dataset and extract test samples
dataset = load_dataset("Vezora/Tested-143k-Python-Alpaca", split="train")
test_subset = dataset.select(range(100000, 101000))  # 1000 samples from remaining data

# Extract prompts
prompts = test_subset["instruction"]
ground_truths = test_subset["output"]

def generate_code(model, tokenizer, prompts, batch_size=8):
    all_outputs = []
    total_batches = len(prompts) // batch_size
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.pad_token_id)

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_outputs.extend(decoded_outputs)

        # Print progress every 10 batches
        if i % (10 * batch_size) == 0:
            print(f"Processed {i // batch_size}/{total_batches} batches")

    return all_outputs

# Generate predictions
original_preds = generate_code(original_model, original_tokenizer, prompts)
fine_tuned_preds = generate_code(fine_tuned_model, fine_tuned_tokenizer, prompts)

weights = (0.25,0.25,0.25,0.25)

og_score = calc_codebleu(ground_truths, original_preds, lang="python", weights=weights)
fine_tuned_score = calc_codebleu(ground_truths, fine_tuned_preds, lang="python", weights=weights)

print(og_score)
print(fine_tuned_score)