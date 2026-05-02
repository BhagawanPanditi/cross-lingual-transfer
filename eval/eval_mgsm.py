import os
import re
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def extract_integer(text):
    match = re.search(r'####(.*)', text, re.DOTALL)
    if match:
        raw_answer = match.group(1)
        raw_answer = raw_answer.replace(',', '')
        nums = re.findall(r'-?\d+', raw_answer)
        if nums:
            return int(nums[0])
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, default="/home/compiling-ganesh/24m0829/forgetting/x_transfer/llama3b")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--langs", type=str, nargs="+", default=["all"], help="Language codes to run (e.g., --langs en fr zh). Default is all.")
    args = parser.parse_args()

    adapter_folder_name = os.path.basename(os.path.normpath(args.adapter_path))
    results_dir = os.path.join("results", adapter_folder_name)
    os.makedirs(results_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, local_files_only=True)
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to(args.device)
    
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    all_input_files = [f for f in os.listdir(args.data_dir) if f.endswith(".jsonl")]
    
    if "all" in args.langs:
        input_files = all_input_files
    else:
        input_files = [f for f in all_input_files if any(f.startswith(f"mgsm_{lang}_") for lang in args.langs)]
        
    print(f"Selected files to process: {input_files}")

    for file_name in input_files:
        input_path = os.path.join(args.data_dir, file_name)
        output_path = os.path.join(results_dir, f"results_{file_name}")
        
        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:
            
            lines = infile.readlines()
            
            for i in tqdm(range(0, len(lines), args.batch_size), desc=file_name):
                batch_lines = lines[i : i + args.batch_size]
                batch_data = [json.loads(line) for line in batch_lines]
                
                prompts = [
                    tokenizer.apply_chat_template(
                        data["messages"],
                        tokenize=False,
                        add_generation_prompt=True
                    ) for data in batch_data
                ]
                
                inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)
                
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=500,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                prompt_length = inputs.input_ids.shape[1]
                
                for j, data in enumerate(batch_data):
                    ground_truth = data["answer"]
                    
                    generated_ids = outputs[j][prompt_length:]
                    generated_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    filtered_answer = extract_integer(generated_answer)
                    
                    result_obj = {
                        "prompt": prompts[j],
                        "ground_truth": ground_truth,
                        "generated_answer": generated_answer,
                        "filtered_answer": filtered_answer
                    }
                    
                    outfile.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
        
if __name__ == "__main__":
    main()
