import os
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm


os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load HumanEval dataset
human_eval = load_dataset("openai_humaneval")['test']

# Load code evaluation metric
code_eval_metric = load("code_eval")

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
original_model.eval()

# Load fine-tuned model
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path, quantization_config=bnb_config, device_map="auto")
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path, padding_side="left")
fine_tuned_tokenizer.pad_token = fine_tuned_tokenizer.eos_token
fine_tuned_model.eval()

# Set the number of candidates per problem
num_samples_per_problem = 5  # Adjust as needed for pass@k computation

# Lists to store test cases and predictions
test_cases = []
og_candidates = []
ft_candidates = []

# Create a progress bar for the outer loop (problems)
print("Generating code solutions...")
for problem in tqdm(human_eval, desc="Problems", unit="problem"):
    prompt = problem['prompt']
    test_code = problem['test']
    # Store the test cases
    test_cases.append(test_code)

    # Generate multiple candidate solutions for each problem
    og_problem_candidates = []
    ft_problem_candidates = []

    # Create a progress bar for the inner loop (samples per problem)
    for _ in range(num_samples_per_problem):
        # Encode the prompt and get attention mask
        og_inputs = original_tokenizer(prompt, return_tensors="pt").to("cuda")
        ft_inputs = fine_tuned_tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generate code with attention mask and proper token IDs
        with torch.no_grad():
            og_outputs = original_model.generate(
                input_ids=og_inputs['input_ids'],
                attention_mask=og_inputs['attention_mask'],
                max_length=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=original_tokenizer.pad_token_id,
                eos_token_id=original_tokenizer.eos_token_id,
            )

            ft_outputs = fine_tuned_model.generate(
                input_ids=ft_inputs['input_ids'],
                attention_mask=ft_inputs['attention_mask'],
                max_length=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=fine_tuned_tokenizer.pad_token_id,
                eos_token_id=fine_tuned_tokenizer.eos_token_id,
            )
        og_generated_code = original_tokenizer.decode(og_outputs[0], skip_special_tokens=True)
        ft_generated_code = fine_tuned_tokenizer.decode(ft_outputs[0], skip_special_tokens=True)
        # Remove the prompt from the generated code
        og_generated_code = og_generated_code[len(prompt):]
        ft_generated_code = ft_generated_code[len(prompt):]

        og_problem_candidates.append(og_generated_code)
        ft_problem_candidates.append(ft_generated_code)

    # Add the candidates for the current problem
    og_candidates.append(og_problem_candidates)
    ft_candidates.append(ft_problem_candidates)

print("Code generation complete.")

# Compute pass@k
k_values = [1, 5]
print("Evaluating generated code...")
pass_at_k, results = code_eval_metric.compute(
    references=test_cases,
    predictions=og_candidates,
    k=k_values,
    num_workers=4,  # Adjust based on your system
    timeout=10.0,   # Adjust the timeout as needed
)

# Print the results
for k in k_values:
    print(f"Original Model: Pass@{k}: {pass_at_k[f'pass@{k}'] * 100:.2f}%")

# Compute pass@k
k_values = [1, 5]
print("Evaluating generated code...")
pass_at_k, results = code_eval_metric.compute(
    references=test_cases,
    predictions=ft_candidates,
    k=k_values,
    num_workers=4,  # Adjust based on your system
    timeout=10.0,   # Adjust the timeout as needed
)

# Print the results
for k in k_values:
    print(f"Fine-tuned Model: Pass@{k}: {pass_at_k[f'pass@{k}'] * 100:.2f}%")