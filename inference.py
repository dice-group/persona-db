import json
import time
from datasets import load_dataset
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "EMPTY"

with open("template.json", "r") as f:
    json_template = json.load(f)
json_template_str = json.dumps(json_template, indent=4)

def build_prompt(persona_text: str, json_template_str: str) -> str:
    field_rules = """
        Field Value Options:
        - ability to speak english: "Fluent", "Moderate", "Beginner", "Novice", "Not Applicable"
        - household language: "English", "Spanish", "Arabic", etc. (can be multiple)
        - big five scores: "Extremely Low", "Low", "Average", "High", "Extremely High"
        - health insurance: "With health insurance coverage", "Without health insurance coverage"
        - class of worker: "Private Organization", "Government", "Self-employed", "Unpaid", "Not applicable"
    """

    dependencies = """
        Field Dependencies:
        - 'ability to speak english' should consider 'citizenship', 'household language', 'education'.
        - 'household language' may reflect multilingualism.
        - 'health insurance' may depend on employment and region.
        - 'big five scores' should reflect cues in the description.
    """

    system_prompt = f"""
    You are an assistant filling in structured JSON persona data from a free-text description.

    Inference:
    - Prefer direct evidence from the persona.
    - If uncertain, use a realistic and consistent value grounded in context.
    - Use "nan" only when nothing reasonable fits.

    Rules:
    - Use valid values only.
    - Keep fields logically consistent (e.g., language ↔ citizenship).
    - Avoid generic or implausible values; favor plausible diversity.
    - Be creative but realistic across different personas.

    {field_rules}
    {dependencies}

    Formatting:
    - Use the template for style, capitalization, and brevity.
    - Match data types (e.g., age: int, income: float).
    - Keep outputs concise and JSON-valid with proper structure.
    """

    user_prompt = (
        f"Given this persona description:\n\n{persona_text}\n\n"
        f"Fill in the following JSON template:\n{json_template_str}\n\n"
        f"Return only the completed JSON. No explanation or extra text."
    )

    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"



def generate_with_vllm(prompt, max_tokens=512):
    response = openai.ChatCompletion.create(
        model="quantized_model_directory",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )
    return response['choices'][0]['message']['content']


dataset = load_dataset("proj-persona/PersonaHub", "persona", split="train[:5]")
prompts = [build_prompt(x["persona"], json_template_str) for x in dataset]

results = []
for prompt in prompts:
    output = generate_with_vllm(prompt)

    try:
        result = json.loads(output.strip())
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        result = {"error": str(e), "raw_output": output}
    
    results.append(result)
    time.sleep(1)

with open("synthetic-persona-template.json", "w") as f:
    json.dump(results, f, indent=4)