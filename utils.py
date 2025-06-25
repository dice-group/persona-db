import json
import os
import re
from typing import Set

def load_json_template(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def format_template_for_prompt(template: dict) -> str:
    return json.dumps({key: "" for key in template.keys()}, indent=2)

def ensure_directory_exists(directory_path: str):
    os.makedirs(directory_path, exist_ok=True)

def get_processed_persona_ids(results_dir: str) -> Set[int]:
    processed_ids = set()
    os.makedirs(results_dir, exist_ok=True)
    
    filename_pattern = re.compile(r"persona_(\d+)(?:_error_gpu\d+)?\.(?:json|txt)")
    
    for filename in os.listdir(results_dir):
        match = filename_pattern.match(filename)
        if match:
            try:
                persona_id = int(match.group(1))
                processed_ids.add(persona_id)
            except ValueError:
                print(f"Warning: Could not parse ID from filename: {filename}")
    return processed_ids

def extract_json_from_output(model_output: str) -> str:
    normalized_marker = re.escape("Only return valid JSON. Do not add explanations or extra text.").replace("\\ ", "\\s*").replace("\\\n", "\\s*")
    pattern = rf"{normalized_marker}\s*(\{{[\s\S]*\}})"
    marker_flexible = r"Only return valid JSON\. Do not add\s*explanations or extra text\.\s*"
    
    match = re.search(marker_flexible, model_output, re.DOTALL | re.IGNORECASE)

    if match:
        substring_after_marker = model_output[match.end():].strip()
        json_start_index = substring_after_marker.find('{')
        
        if json_start_index != -1:
            try:
                potential_json_str = substring_after_marker[json_start_index:]
                
                brace_count = 0
                json_end_index = -1
                
                for i, char in enumerate(potential_json_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end_index = i
                            break
                
                if json_end_index != -1:
                    return potential_json_str[:json_end_index + 1]
                else:
                    print("Warning: Found marker but JSON after it was incomplete or malformed (brace mismatch).")
                    return ""
            except Exception as e:
                print(f"Error during post-regex JSON parsing: {e}")
                return ""
        else:
            print("Warning: Marker found, but no opening brace '{' immediately followed.")
            return ""
    else:
        print("Warning: Specific marker 'Only return valid JSON. Do not add explanations or extra text.' not found in output.")

        potential_jsons = []
        brace_count = 0
        json_start_index = -1
        
        for i, char in enumerate(model_output):
            if char == '{':
                brace_count += 1
                if json_start_index == -1:
                    json_start_index = i
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start_index != -1:
                    potential_jsons.append(model_output[json_start_index : i + 1])
                    json_start_index = -1 # Reset for next potential JSON
        
        if potential_jsons:
            return potential_jsons[-1]
        else:
            return ""