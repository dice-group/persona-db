import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

token = os.environ["HF_TOKEN"]

model_path = 'meta-llama/Llama-3.3-70B-Instruct'
quant_path = model_path + "_awq"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
# model = AutoAWQForCausalLM.from_pretrained(
#     model_path, **{"low_cpu_mem_usage": True, "use_cache": False, "device_map": "auto"} 
# )
model = AutoAWQForCausalLM.from_pretrained(
    model_path, use_cache=False, device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=token)


# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# # Save quantized model
model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')

