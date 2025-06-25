import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Union


class LlamaModel:
    def __init__(self, model_name: str) -> None:
        """
        Initialize the LlamaModel with a specified model name.

        Args:
            model_name (str): The name or path of the pretrained model to load.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

        self.model.eval()

    def generate_response(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> Union[str, List[str]]:
        """
        Generate text responses for one or more prompts.

        Args:
            prompts (Union[str, List[str]]): A single prompt string or a list of prompt strings.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 1024.
            temperature (float, optional): Sampling temperature. Higher values increase randomness. Defaults to 0.2.

        Returns:
            Union[str, List[str]]: Generated response(s) as a string or list of strings.
        """
        if isinstance(prompts, str):
            prompts_list = [prompts]
            return_single = True
        else:
            prompts_list = prompts
            return_single = False

        inputs = self.tokenizer(
            prompts_list, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0.0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        decoded_outputs = []
        for i in range(len(prompts_list)):
            output_sequence = outputs[i][inputs["input_ids"].shape[1] :]
            decoded_text = self.tokenizer.decode(
                output_sequence, skip_special_tokens=True
            )
            decoded_outputs.append(decoded_text)

        if return_single:
            return decoded_outputs[0]
        else:
            return decoded_outputs
