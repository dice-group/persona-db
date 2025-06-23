import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

model_name = "meta-llama/Llama-3.3-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

datasets = load_dataset("proj-persona/PersonaHub", "persona", split="train[:5]")

def load_template(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def format_template_for_prompt(template: dict) -> str:
    return json.dumps({key: "" for key in template.keys()}, indent=2)

def build_prompt(persona: str, template_json: dict) -> str:
    template_str = format_template_for_prompt(template_json)

    field_rules = """
    Field Value Options:
    - ability to speak english: "Fluent", "Moderate", "Beginner", "Novice", "Not Applicable"
    - household language: "English", "Spanish", "Arabic", etc. (can be multiple)
    - each of 'Openness', 'Conscientiousness', 'Extraversion', ' Agreeableness' and 'Neuroticism': "Extremely Low", "Low", "Average", "High", "Extremely High"
    - health insurance: "With health insurance coverage", "Without health insurance coverage"
    - class of worker: "Private Organization", "Government", "Self-employed", "Unpaid", "Not applicable", "Public Sector" etc..
    """.strip()

    dependencies = """
    Field Dependencies:
    - 'ability to speak english' should consider 'citizenship', 'household language', 'education'.
    - 'household language' may reflect multilingualism.
    - 'health insurance' may depend on employment and region.
    """

    prompt = f"""
    You are a helpful assistant that extracts structured information from personas.

    Given the following persona:
    \"\"\"{persona}\"\"\"

    Instructions:
    - Fill in the following JSON fields based only on the persona information.
    - If something is not specified or uncertain, creatively fill up with what is appropriate and realistic to the persona.
    - Avoid generic or implausible values but be a creative and realistic across different personas.
    - Use the following value constraints when applicable and DO NOT collapse nested keys in JSON template.

    {field_rules}

    {dependencies}

    - Here's the JSON template to be considered but DO NOT output it in the generated output.
    {template_str}

    Only return valid JSON. Do not add explanations or extra text.
    """.strip()
    return prompt

def run_model(prompt: str, tokenizer, model, max_new_tokens: int = 1024) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

template_json = load_template("template.json")
prompts = [build_prompt(data["persona"], template_json) for data in datasets]

for prompt in prompts:
    generated = run_model(prompt, tokenizer, model)
    print(generated)
    print('='*100)
    
