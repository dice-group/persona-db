import json
import os
import re
from typing import Set

def load_json_template(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def format_template_for_prompt(template: dict) -> str:
    return json.dumps({key: "" for key in template.keys()}, indent=2)

def ensure_directory_exists(directory_path: str):
    os.makedirs(directory_path, exist_ok=True)

def get_processed_persona_ids(results_dir: str) -> Set[int]:
    processed_ids = set()
    os.makedirs(results_dir, exist_ok=True)

    json_filename_pattern = re.compile(r"persona_(\d+)\.json")
    error_filename_pattern = re.compile(
        r"persona_\d+_error(?:_gpu\d+)?(?:_OOM|_no_output)?\.txt"
    )

    if not hasattr(get_processed_persona_ids, "_deleted_errors_this_run"):
        get_processed_persona_ids._deleted_errors_this_run = False

    if not get_processed_persona_ids._deleted_errors_this_run:
        files_to_delete = []
        for filename in os.listdir(results_dir):
            error_match = error_filename_pattern.match(filename)
            if error_match:
                files_to_delete.append(os.path.join(results_dir, filename))

        for filepath in files_to_delete:
            try:
                os.remove(filepath)
            except OSError:
                pass

        get_processed_persona_ids._deleted_errors_this_run = True

    for filename in os.listdir(results_dir):
        json_match = json_filename_pattern.match(filename)
        if json_match:
            try:
                persona_id = int(json_match.group(1))
                processed_ids.add(persona_id)
            except ValueError:
                pass

    return processed_ids

def extract_json_from_output(text: str) -> str:
    extracted_json_str = ""
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    
    if json_match:
        extracted_json_str = json_match.group(1)
        
        try:
            parsed_json = json.loads(extracted_json_str)
            
            if isinstance(parsed_json, dict) and len(parsed_json) == 33:
                return extracted_json_str
            else:
                pass 
        except json.JSONDecodeError:
            pass

    first_brace_idx = text.find('{')
    if first_brace_idx != -1:
        decoder = json.JSONDecoder()
        try:
            parsed_json, end_idx = decoder.raw_decode(text[first_brace_idx:])
            fallback_json_str = text[first_brace_idx : first_brace_idx + end_idx]
            
            if isinstance(parsed_json, dict) and len(parsed_json) == 33:
                return fallback_json_str
            else:
                pass 
        except json.JSONDecodeError:
            pass

    return text