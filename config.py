MODEL_NAME = (
    "../quantize/Llama-3.3-70B-Instruct_awq"  # Ensure it matches with your directory
)
DATASET_NAME = "proj-persona/PersonaHub"
DATASET_SUBSET = "persona"
DATASET_SPLIT = "train[:5001]"
TEMPLATE_PATH = "template.json"
RESULTS_DIR = "results"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
MAX_NUM_SEQS = 128
