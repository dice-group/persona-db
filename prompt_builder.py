import json


def build_extraction_prompt(persona: str, template_json: dict) -> str:
    """
    Build a prompt for extracting structured information from a persona using a JSON template.

    Args:
        persona (str): The text content of the persona.
        template_json (dict): The JSON template as a dictionary.

    Returns:
        str: The formatted prompt string for the language model.
    """
    template_str = json.dumps({key: "" for key in template_json.keys()}, indent=2)

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
    - If something is not specified or uncertain, creatively fill in with what is appropriate and realistic to the persona.
    - Avoid generic or implausible values, but be creative and realistic across different personas.
    - Not all Europeans or non-Europeans who are residents (citizens) of the United States (United Kingdom) are fluent in English, though the majority are.
    - For inclusion and diversity of personas, seldom fill 'disability', 'vision difficulty', and 'veteran status' with plausible values (instead of 'None') correlating to other keys in the JSON template
    - Use the following value constraints when applicable and DO NOT collapse nested keys in the JSON template.
    - DO NOT generate any further text or tokens in any format again.

    Formatting:
    - Use the template for style, capitalization, and brevity.
    - Match data types (e.g., age: int, income: float).
    - Keep outputs concise and JSON-valid with proper structure where applicable.

    {field_rules}

    {dependencies}

    - Here's the JSON template to be considered, but DO NOT output it in the generated output.
    {template_str}

    - Once the JSON template has been filled, start its output with "```json" and end it with "```", then DO NOT generate any further text or tokens in any format again. 
    """.strip()
    return prompt
