import os
import json
import copy
import time
import argparse
import logging
from typing import Any, Dict, List
import random
import numpy as np
import torch
import tqdm
from datasets import Dataset
from visualization import analyze_debug_infos
# Assume eval.py is in the same directory and defines the `Model` class
from eval import Model

logger = logging.getLogger(__name__)

def parse_unknown_args(unknown: List[str]) -> Dict[str, Any]:
    """Parse arbitrary --key value pairs from CLI."""
    if len(unknown) % 2 != 0:
        raise ValueError("All extra arguments must be in --key value pairs.")
    kwargs = {}
    for i in range(0, len(unknown), 2):
        key = unknown[i]
        if not key.startswith("--"):
            raise ValueError(f"Expected argument to start with '--', got: {key}")
        key = key[2:]  # remove '--'
        value = unknown[i + 1]

        # Try to auto-convert types
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # keep as string
        kwargs[key] = value
    return kwargs


def main():
    parser = argparse.ArgumentParser(description="DAEDAL-style evaluation with lm-eval Model backend.")
    parser.add_argument("--cuda-i", type=int, required=True, help="Local GPU index (e.g., 0). Only one allowed.")

    known_args, unknown_args = parser.parse_known_args()
    extra_kwargs = parse_unknown_args(unknown_args)

    # --- Environment-based multi-GPU setup ---
    split_num = os.environ.get('WORLD_SIZE', 1)
    now_i = os.environ.get("RANK", 0)
    seed = int(os.environ.get("SEED", 0))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Data paths
    dataset_local_path = 'Annoymous for review'
    local_save = 'Annoymous for review'

    # Load dataset
    with open(dataset_local_path, "r", encoding="utf-8") as f:
        data_list = [json.loads(line.strip()) for line in f if line.strip()]

    chunk_size = -(-len(data_list) // split_num)
    data_shard = data_list[now_i * chunk_size : (now_i + 1) * chunk_size]

    if not data_shard:
        logger.warning(f"Worker {now_i} has no data. Exiting.")
        return

    model_instance = Model(**extra_kwargs)
    print(f"Successfully load data length {len(data_shard)}")
    print(f"extra_kwargs: {extra_kwargs}")
    print(f"model: {type(model_instance)}")

    # Prepare prompts
    prompts = []
    original_datas = []
    for data in data_shard:
        prompt = data["prompt"]
        messages = [{"role": "user", "content": prompt}]
        text = model_instance.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
        original_datas.append(data)

    # === BEGIN: Custom generation loop with timing & token counting ===
    total_gen_tokens = 0
    start_time = time.time()

    results = []
    debug_list = []
    batch_size = getattr(model_instance, 'batch_size', 1)

    for i in tqdm.tqdm(range(0, len(prompts), batch_size), desc=f"cuda_{known_args.cuda_i}"):
        batch_prompts = prompts[i:i + batch_size]
        batch_original = original_datas[i:i + batch_size]

        # Tokenize
        context_enc, attn_masks = model_instance.tok_batch_encode(batch_prompts)
        prompt_len = context_enc.shape[1]

        # Handle accidental mask_id in prompt (same as original generate_until)
        if (context_enc == model_instance.config.mask_id).any():
            another_token = model_instance.tokenizer.encode(' ')[0]
            if another_token == model_instance.config.mask_id:
                another_token = model_instance.tokenizer.encode('?')[0]
            context_enc[context_enc == model_instance.config.mask_id] = another_token

        # Generate
        with torch.no_grad():
            output_ids, dbg = model_instance.gen_func(input_ids=context_enc)
            debug_list.append(dbg)

        # Decode only the generated part
        gen_part = output_ids[:, prompt_len:]
        cont_toks_list = model_instance.tokenizer.batch_decode(gen_part, skip_special_tokens=True)

        # Count generated tokens (non-padding, but since we skip special tokens and use full output, .numel() is fine)
        total_gen_tokens += gen_part.numel()

        # Post-process each result
        for gen, orig in zip(cont_toks_list, batch_original):
            if "[unused16]" in gen and "[unused17]" in gen:
                thinking = gen.split("[unused16]")[-1].split("[unused17]")[0].strip()
                response = gen.split("[unused17]")[-1].split("[unused10]")[0].strip()
            else:
                thinking = ""
                response = gen.strip()

            result_data = copy.deepcopy(orig)
            result_data["thinking_content"] = thinking
            result_data["response"] = response
            results.append(result_data)

            print(f"## --- cuda_{known_args.cuda_i}:\nthinking:\n{thinking}\nresponse:\n{response}\n")

    end_time = time.time()
    total_time = end_time - start_time

    # Save main results (same as before)
    datasets_name = os.path.basename(dataset_local_path).split(".")[0]
    os.makedirs(local_save, exist_ok=True)
    output_path = f"{local_save}/{datasets_name}_{now_i}.json"
    with open(output_path, "w", encoding="utf-8") as outf:
        for item in results:
            outf.write(json.dumps(item, ensure_ascii=False) + "\n")

    # === Save statistics ===
    stat_path = f"{local_save}/{datasets_name}_{now_i}_stat"
    stat_data = {
        "time": round(total_time, 3),
        "tokens": total_gen_tokens
    }
    with open(stat_path, "w", encoding="utf-8") as sf:
        json.dump(stat_data, sf, ensure_ascii=False, indent=2)

    debug_path = f"{local_save}/{datasets_name}_{now_i}_dbg"
    if model_instance.debug_type in ['full', 'stat']:
        analyze_debug_infos(debug_list, debug_path)

    logger.info(f"Saved {len(results)} results to {output_path}")
    logger.info(f"Saved stats (time={total_time:.3f}s, tokens={total_gen_tokens}) to {stat_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()