import os
import json
import re
import time
import torch
import multiprocessing as mp
from datasets import load_dataset
from config import (
    MODEL_NAME,
    DATASET_NAME,
    DATASET_SUBSET,
    DATASET_SPLIT,
    TEMPLATE_PATH,
    RESULTS_DIR,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    BATCH_SIZE,
)
from utils import (
    load_json_template,
    ensure_directory_exists,
    extract_json_from_output,
    get_processed_persona_ids,
)
from prompt_builder import build_extraction_prompt
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


def _worker_process_wrapper(args):
    return worker_process(*args)


def worker_process(
    persona_data_chunk: List[Tuple[int, str]],
    gpu_id: int,
    config_params: Dict[str, Any],
) -> List[Tuple[int, str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from inference import LlamaModel

    model_name = config_params["MODEL_NAME"]
    template_path = config_params["TEMPLATE_PATH"]
    results_dir = config_params["RESULTS_DIR"]
    max_new_tokens = config_params["MAX_NEW_TOKENS"]
    temperature = config_params["TEMPERATURE"]
    batch_size = config_params["BATCH_SIZE"]

    results_for_main_process = []

    try:
        llama_model = LlamaModel(model_name)
        template_json = json.loads(open(template_path, "r").read())
        os.makedirs(results_dir, exist_ok=True)

        prompts_to_process = []
        original_indices_map = {}

        for local_idx, (global_idx, persona_text) in enumerate(persona_data_chunk):
            prompt = build_extraction_prompt(persona_text, template_json)
            prompts_to_process.append(prompt)
            original_indices_map[local_idx] = global_idx

        num_personas_in_chunk = len(prompts_to_process)

        for i in range(0, num_personas_in_chunk, batch_size):
            batch_prompts = prompts_to_process[i : i + batch_size]

            try:
                full_model_outputs: List[str] = llama_model.generate_response(
                    batch_prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except torch.cuda.OutOfMemoryError as e:
                print(
                    f"!!! Process on GPU {gpu_id}: CUDA Out of Memory for batch starting with persona {original_indices_map[i // batch_size * batch_size]}. Skipping batch. Error: {e}"
                )
                for j in range(len(batch_prompts)):
                    local_batch_idx = i + j
                    persona_global_idx = original_indices_map[local_batch_idx]
                    filename = os.path.join(
                        results_dir,
                        f"persona_{persona_global_idx}_error_gpu{gpu_id}_OOM.txt",
                    )
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(
                            f"CUDA Out Of Memory Error during processing. Original prompt:\n{batch_prompts[j]}\nError: {e}"
                        )
                    results_for_main_process.append(
                        (persona_global_idx, "cuda_oom_error", str(e))
                    )
                continue

            for j, output_text in enumerate(full_model_outputs):
                local_batch_idx = i + j
                persona_global_idx = original_indices_map[local_batch_idx]

                extracted_json_str = extract_json_from_output(output_text)

                try:
                    parsed_json = json.loads(extracted_json_str)

                    filename = os.path.join(
                        results_dir, f"persona_{persona_global_idx}.json"
                    )
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(parsed_json, f, indent=2, ensure_ascii=False)

                    results_for_main_process.append(
                        (persona_global_idx, "success", parsed_json)
                    )

                except json.JSONDecodeError as e:
                    filename = os.path.join(
                        results_dir,
                        f"persona_{persona_global_idx}_error_gpu{gpu_id}.txt",
                    )
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(output_text)
                    results_for_main_process.append(
                        (persona_global_idx, "json_error", str(e))
                    )
                except Exception as e:
                    filename = os.path.join(
                        results_dir,
                        f"persona_{persona_global_idx}_error_gpu{gpu_id}.txt",
                    )
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(output_text)
                    results_for_main_process.append(
                        (persona_global_idx, "other_error", str(e))
                    )

        return results_for_main_process

    except Exception as e:
        print(
            f"!!! Process on GPU {gpu_id}: A critical error occurred in worker process: {e}"
        )
        import traceback

        traceback.print_exc()
        error_results = []
        for global_idx, _ in persona_data_chunk:
            error_results.append((global_idx, "critical_error", str(e)))
        return error_results


def main_multi_gpu():
    mp.set_start_method("spawn", force=True)
    print("Multiprocessing start method set to 'spawn'.")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print(
            "No GPUs found. Please check your CUDA installation and drivers. Exiting."
        )
        return

    print(f"Found {num_gpus} GPUs. Preparing to distribute workload...")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Main process: Ensured results directory exists: {RESULTS_DIR}")

    print(
        f"Main process: Loading full dataset: {DATASET_NAME}/{DATASET_SUBSET} split {DATASET_SPLIT}..."
    )
    full_datasets = load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)
    total_personas_in_dataset = len(full_datasets)
    print(
        f"Main process: Full dataset loaded. Total personas in dataset: {total_personas_in_dataset}"
    )

    processed_ids = get_processed_persona_ids(RESULTS_DIR)

    personas_to_process_tuples = []
    for global_idx, data_entry in enumerate(full_datasets):
        if global_idx not in processed_ids:
            personas_to_process_tuples.append((global_idx, data_entry["persona"]))

    remaining_personas_count = len(personas_to_process_tuples)

    print(
        f"\n--- Progress Summary: {total_personas_in_dataset - remaining_personas_count} completed, {remaining_personas_count} left out of {total_personas_in_dataset} total personas ---\n"
    )

    if remaining_personas_count == 0:
        print("All personas have been processed. Exiting.")
        return

    chunked_data_with_indices: List[List[Tuple[int, str]]] = [
        [] for _ in range(num_gpus)
    ]
    for i, (global_idx, persona_text) in enumerate(personas_to_process_tuples):
        target_gpu_idx = i % num_gpus
        chunked_data_with_indices[target_gpu_idx].append((global_idx, persona_text))

    config_for_workers = {
        "MODEL_NAME": MODEL_NAME,
        "TEMPLATE_PATH": TEMPLATE_PATH,
        "RESULTS_DIR": RESULTS_DIR,
        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "TEMPERATURE": TEMPERATURE,
        "BATCH_SIZE": BATCH_SIZE,
    }

    worker_args = []
    for i in range(num_gpus):
        if chunked_data_with_indices[i]:
            worker_args.append((chunked_data_with_indices[i], i, config_for_workers))

    start_full_process_time = time.time()

    with mp.Pool(processes=num_gpus) as pool:
        for _ in tqdm(
            pool.imap_unordered(_worker_process_wrapper, worker_args),
            total=len(worker_args),
            desc="Processing Chunks by GPU",
        ):
            pass

    end_full_process_time = time.time()
    final_total_time = end_full_process_time - start_full_process_time

    print("\n--- All GPU workers finished processing. ---")
    print(
        f"Overall total processing time for remaining {remaining_personas_count} personas: {final_total_time:.2f} seconds"
    )
    if remaining_personas_count > 0:
        print(
            f"Average processing time per remaining persona: {final_total_time / remaining_personas_count:.2f} seconds"
        )


if __name__ == "__main__":
    main_multi_gpu()
