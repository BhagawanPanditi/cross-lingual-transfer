import json
import os
from datasets import load_dataset

def process_mgsm():
    languages = ["zh", "ja", "bn", "sw", "ru", "de", "es", "fr", "te", "th", "en"]
    
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    for lang in languages:        
        ds = load_dataset("juletxara/mgsm", lang, trust_remote_code=True)
        output_file = os.path.join(output_dir, f"mgsm_{lang}_test.jsonl")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for item in ds["test"]:
                raw_question = item["question"]
                ans_num = item["answer_number"]
                
                json_obj = {
                    "messages": [
                        {
                            "role": "user",
                            "content": raw_question
                        }
                    ],
                    "answer": ans_num
                }
                
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        
if __name__ == "__main__":
    process_mgsm()
