import os
import json
import glob
import csv

def main():
    base_dir = "/home/compiling-ganesh/24m0829/forgetting/x_transfer/eval/results"
    
    csv_data = {}
    lambda_columns = []

    for subdir_name in os.listdir(base_dir):
        if not subdir_name.startswith("output_lambda_"):
            continue

        subdir_path = os.path.join(base_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue
            
        jsonl_files = glob.glob(os.path.join(subdir_path, "results_mgsm_*_test.jsonl"))
        if not jsonl_files:
            continue
            
        print(f"📊 Processing folder: {subdir_name}")
        
        lambda_val = subdir_name.replace("output_lambda_fixed_seed_", "")
        if lambda_val not in lambda_columns:
            lambda_columns.append(lambda_val)

        summary = {}
        overall_correct = 0
        overall_total = 0

        for file_path in jsonl_files:
            filename = os.path.basename(file_path)
            lang = filename.split('_')[2]
            
            lang_correct = 0
            lang_total = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                        
                    data = json.loads(line)
                    lang_total += 1
                    
                    if data.get("ground_truth") == data.get("filtered_answer"):
                        lang_correct += 1
            
            accuracy = (lang_correct / lang_total) * 100 if lang_total > 0 else 0.0
            rounded_acc = round(accuracy, 2)
            
            summary[lang] = {
                "correct": lang_correct,
                "total": lang_total,
                "accuracy": rounded_acc
            }
            
            if lang not in csv_data:
                csv_data[lang] = {}
            csv_data[lang][lambda_val] = rounded_acc
            
            overall_correct += lang_correct
            overall_total += lang_total
            
        overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0.0
        rounded_overall = round(overall_accuracy, 2)
        
        summary["overall"] = {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": rounded_overall
        }
        
        if "overall" not in csv_data:
            csv_data["overall"] = {}
        csv_data["overall"][lambda_val] = rounded_overall
        
        summary_path = os.path.join(subdir_path, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as out_file:
            json.dump(summary, out_file, indent=4, ensure_ascii=False)

    if not csv_data:
        print("❌ No data found to process.")
        return

    lambda_columns.sort()
    
    languages = list(csv_data.keys())
    if "overall" in languages:
        languages.remove("overall")
    languages.sort()
    languages.append("overall")

    csv_file_path = os.path.join(base_dir, "lambda_comparison.csv")
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        header = ["Language"] + lambda_columns
        writer.writerow(header)
        
        for lang in languages:
            row = [lang]
            for l in lambda_columns:
                row.append(csv_data[lang].get(l, ""))
            writer.writerow(row)
            
    print(f"✅ Success! Summaries generated and Comparison CSV saved at: {csv_file_path}")

if __name__ == "__main__":
    main()
