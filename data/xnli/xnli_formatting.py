import json
import re

input_file = 'xnli_validation.jsonl'
output_file = 'xnli_validation_formatted.jsonl'

# Mapping the conversational labels to standard Academic labels
label_map = {
    "True": "Entailment",
    "False": "Contradiction",
    "Neither": "Neutral"
}

# The instruction that will be baked into the user prompt
instruction = "Determine the relationship between the premise and the hypothesis. Respond with only one word: Entailment, Contradiction, or Neutral."

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        if not line.strip():
            continue
            
        data = json.loads(line.strip())
        
        user_msg = data['messages'][0]['content']
        assistant_msg = data['messages'][1]['content']
        
        # 1. Split the premise and hypothesis based on your dataset's text pattern
        parts = user_msg.split('\nQuestion:')
        if len(parts) != 2:
            print(f"Skipping malformed line: {user_msg}")
            continue 
            
        premise = parts[0].strip()
        hypothesis_raw = parts[1]
        
        # 2. Clean up the hypothesis
        # Remove the 'True, False, or Neither?' text using regex
        hypothesis = re.sub(r'True,\s*False,\s*or\s*Neither\?$', '', hypothesis_raw).strip()
        
        # Strip trailing/leading quotes if they exist in the raw string
        hypothesis = hypothesis.strip(' "')
        
        # 3. Format the new user message (Instruction + Premise + Hypothesis)
        new_user_content = f"{instruction}\n\nPremise: {premise}\nHypothesis: {hypothesis}"
        
        # 4. Map the target label to the new format
        target_label = assistant_msg.strip()
        new_label = label_map.get(target_label, target_label)
        
        # 5. Overwrite the dictionary values
        data['messages'][0]['content'] = new_user_content
        data['messages'][1]['content'] = new_label
        
        # 6. Write back out to the new JSONL file
        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"Conversion complete! Saved to {output_file}")
