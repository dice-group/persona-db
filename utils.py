import json
import os
import re
from typing import Set, Dict


def load_json_template(path: str) -> dict:
    """
    Load a JSON template from a file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict: Loaded JSON as a dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)


def format_template_for_prompt(template: dict) -> str:
    """
    Format a JSON template for use in a prompt, setting all values to empty strings.

    Args:
        template (dict): The JSON template.

    Returns:
        str: JSON string with all values set to empty strings.
    """
    return json.dumps({key: "" for key in template.keys()}, indent=2)

def assert_file_exists(file_path: str):
    """Raises AssertionError if the file does not exist."""
    assert os.path.isfile(file_path), f"File does not exist: {file_path}"
    
def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory_path (str): Path to the directory.
    """
    os.makedirs(directory_path, exist_ok=True)


def get_processed_persona_ids(
    results_dir: str, expected_global_ids_in_split: Set[int]
) -> Set[int]:
    """
    Get the set of processed persona IDs by scanning the results directory for JSON files.

    Args:
        results_dir (str): Directory containing result files.
        expected_global_ids_in_split (Set[int]): Set of expected global persona IDs in the current dataset split.

    Returns:
        Set[int]: Set of processed persona IDs found in the directory and present in the expected IDs.
    """
    processed_ids = set()
    if not os.path.exists(results_dir):
        return processed_ids

    for filename in os.listdir(results_dir):
        if filename.startswith("persona_") and filename.endswith(".json"):
            try:
                match = re.match(r"persona_(\d+)\.json$", filename)
                if match:
                    persona_id = int(match.group(1))
                    if persona_id in expected_global_ids_in_split:
                        processed_ids.add(persona_id)
            except ValueError:
                continue
    return processed_ids


def extract_json_from_output(text: str) -> str:
    """
    Extract a JSON string from model output text.

    Args:
        text (str): The output text from the model.

    Returns:
        str: Extracted JSON string if found, otherwise returns the original text.
    """
    extracted_json_str = ""
    # json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_match = re.search(r"```json\s*(.*?)(?:```|(?=\s*[}\]]?\s*$))", text, re.DOTALL)

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

    first_brace_idx = text.find("{")
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
