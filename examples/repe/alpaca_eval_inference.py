import os
import json
from typing import List

import numpy as np
import yaml
import argparse
import datasets
from concurrent.futures import ThreadPoolExecutor
import subprocess
import time

from tqdm import tqdm

from jailbreakpipe.llms import create_llm, LLMGenerateConfig
from jailbreakpipe.llms.repe import RepeLLM, RepeLLMConfig
from jailbreakpipe.utils import wait_for_gpu_memory


# Load the YAML configuration file
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


# python alpaca_eval_inference.py --target-llm Qwen/Qwen2-7B-Instruct --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:1 --batch-size 20
# python alpaca_eval_inference.py --target-llm meta-llama/Meta-Llama-3-8B-Instruct --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:2 --batch-size 20
# python alpaca_eval_inference.py --target-llm mistralai/Mistral-7B-Instruct-v0.3 --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:3 --batch-size 20
# python alpaca_eval_inference.py --target-llm microsoft/Phi-3-mini-4k-instruct --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:4 --batch-size 20
# python alpaca_eval_inference.py --target-llm Qwen/Qwen1.5-7B-Chat --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:5 --batch-size 20
# python alpaca_eval_inference.py --target-llm meta-llama/Meta-Llama-3.1-8B-Instruct --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:6 --batch-size 20
# python alpaca_eval_inference.py --target-llm google/gemma-2-9b-it --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:7 --batch-size 20


# python alpaca_eval_inference.py --target-llm google/gemma-2-9b-it --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --split 0 --device cuda:0 --batch-size 5
# python alpaca_eval_inference.py --target-llm google/gemma-2-9b-it --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --split 1 --device cuda:1 --batch-size 5
# python alpaca_eval_inference.py --target-llm google/gemma-2-9b-it --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --split 2 --device cuda:2 --batch-size 5
# python alpaca_eval_inference.py --target-llm google/gemma-2-9b-it --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --split 3 --device cuda:3 --batch-size 5
# python alpaca_eval_inference.py --target-llm google/gemma-2-9b-it --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --split 4 --device cuda:4 --batch-size 5
# python alpaca_eval_inference.py --target-llm google/gemma-2-9b-it --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --split 5 --device cuda:5 --batch-size 5
# python alpaca_eval_inference.py --target-llm google/gemma-2-9b-it --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --split 6 --device cuda:6 --batch-size 5
# python alpaca_eval_inference.py --target-llm google/gemma-2-9b-it --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --split 7 --device cuda:7 --batch-size 5

# python alpaca_eval_inference.py --target-llm meta-llama/Meta-Llama-3-70B-Instruct --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device auto --batch-size 100

# python alpaca_eval_inference.py --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:0 --batch-size 10 --split 1 --factor 0.333 --target-llm Qwen/Qwen2-7B-Instruct
# python alpaca_eval_inference.py --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:1 --batch-size 10 --split 1 --factor 0.333 --target-llm meta-llama/Meta-Llama-3-8B-Instruct
# python alpaca_eval_inference.py --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:2 --batch-size 10 --split 1 --factor 0.333 --target-llm microsoft/Phi-3-mini-4k-instruct
# python alpaca_eval_inference.py --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:3 --batch-size 10 --split 1 --factor 0.333 --target-llm Qwen/Qwen1.5-7B-Chat
# python alpaca_eval_inference.py --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:4 --batch-size 10 --split 1 --factor 0.333 --target-llm meta-llama/Meta-Llama-3.1-8B-Instruct
# python alpaca_eval_inference.py --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:5 --batch-size 5 --split 1 --factor 0.333 --target-llm google/gemma-2-9b-it
# python alpaca_eval_inference.py --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:6 --batch-size 10 --split 1 --factor 0.333 --target-llm google/gemma-2-2b-it
# python alpaca_eval_inference.py --llms ../../configs/repe/llms_all_layers.yaml --output-dir ../../results/alpaca_eval/repe_.9 --device cuda:7 --batch-size 10 --split 1 --target-llm google/gemma-2-2b-it


# python alpaca_eval_inference.py --target-llm google/gemma-2-2b-it --llms ../../configs/repe/sparse.yaml --output-dir ../../results/alpaca_eval/sparsity --split 0 --total 8 --device cuda:7 --batch-size 5
# python alpaca_eval_inference.py --target-llm google/gemma-2-2b-it --llms ../../configs/repe/sparse.yaml --output-dir ../../results/alpaca_eval/sparsity --split 1 --total 8 --device cuda:6 --batch-size 5
# python alpaca_eval_inference.py --target-llm google/gemma-2-2b-it --llms ../../configs/repe/sparse.yaml --output-dir ../../results/alpaca_eval/sparsity --split 2 --total 8 --device cuda:5 --batch-size 5
# python alpaca_eval_inference.py --target-llm google/gemma-2-2b-it --llms ../../configs/repe/sparse.yaml --output-dir ../../results/alpaca_eval/sparsity --split 3 --total 8 --device cuda:4 --batch-size 5
# python alpaca_eval_inference.py --target-llm google/gemma-2-2b-it --llms ../../configs/repe/sparse.yaml --output-dir ../../results/alpaca_eval/sparsity --split 4 --total 8 --device cuda:3 --batch-size 5
# python alpaca_eval_inference.py --target-llm google/gemma-2-2b-it --llms ../../configs/repe/sparse.yaml --output-dir ../../results/alpaca_eval/sparsity --split 5 --total 8 --device cuda:2 --batch-size 5
# python alpaca_eval_inference.py --target-llm google/gemma-2-2b-it --llms ../../configs/repe/sparse.yaml --output-dir ../../results/alpaca_eval/sparsity --split 5 --total 8 --device cuda:1 --batch-size 5
# python alpaca_eval_inference.py --target-llm google/gemma-2-2b-it --llms ../../configs/repe/sparse.yaml --output-dir ../../results/alpaca_eval/sparsity --split 5 --total 8 --device cuda:0 --batch-size 5


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument('--target-llm', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--llms', type=str, default="../../configs/repe/llms.yaml", help='Path to list of available LLMs')
    parser.add_argument('--llm-gen', type=str, default="../../configs/repe/llm_gen.yaml", help='Path to list of available LLMs')
    parser.add_argument('--output-dir', type=str, default="../../results/alpaca_eval/repe", help='Output directory')
    parser.add_argument('--ctrl-split', type=int, default=20, help='Number of times to repeat the experiment')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for generation')
    parser.add_argument('--device', type=str, default='cuda:0', help='Number of GPUs to use')
    parser.add_argument('--split', default=0, type=int)
    parser.add_argument('--total', default=1, type=int)

    parser.add_argument('--factor', default=1, type=float)
    return parser.parse_args()


# Parallel inference function for each model
def run_model_inference(model_name, model_value, eval_set, llm_gen_config, ctrl_factors, topk, selector, batch_size, output_base_dir, device_id):
    device_map = device_id  # Assign each model to a specific GPU
    # if model_value['device_map'] is None:
    model_value['device_map'] = device_map  # Update device_map in the model config
    model_value['ctrl_factor'] = 0.0  # Set the control factor to 0.0
    model_value['topk'] = 0  # Set the topk to 0
    model_value['selector'] = 'abs_max'

    llm_config = RepeLLMConfig(**model_value)
    print(llm_config)
    llm = create_llm(llm_config)
    print(llm_config)

    if topk is not None:
        if isinstance(topk, float):
            topk = [topk]
    else:
        topk = [llm.topk]
    # For each control factor

    for select in selector:
        for tk in topk:
            for ctrl_factor in ctrl_factors:
                results = []

                output_file = os.path.join(output_base_dir.format(topk=f'{select}/{tk:.3f}'), f'full_{ctrl_factor:.3f}.json')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                if os.path.exists(output_file):
                    print(f"File {output_file} already exists. Skipping...")
                    continue  # Skip if file already exists
                print(output_file)
                # Set the LLM activation according to the control factor
                llm.set_activations(ctrl_factor, tk, select)

                # Process evaluation set in batches
                for i in tqdm(range(0, len(eval_set), batch_size), desc=f"Running {model_name}, ctrl_factor={ctrl_factor:.3f}"):
                    # print(eval_set[i: i+3], batch_size)
                    instructions = eval_set[i: i + batch_size]['instruction']
                    # print(instructions)

                    # Prepare message format for LLM
                    if 'gemma' not in llm_config.model_name:
                        messages = [[{"role": "system", "content": "You are a helpful assistant."},
                                     {"role": "user", "content": instruction}]
                                    for instruction in instructions]
                    else:
                        messages = [[{"role": "user", "content": instruction}] for instruction in instructions]

                    # Generate LLM responses
                    generated_results = llm.batch_generate(messages, llm_gen_config)

                    # Extract content from the last message and update eval_set
                    for idx, result in enumerate(generated_results):
                        output_content = result[-1]['content']  # Assuming result contains 'content'
                        row = eval_set[i + idx]  # Copy original data from eval_set
                        row["output"] = output_content  # Add output to the row
                        row["generator"] = f'{model_name.replace("/", "_")}_full_{ctrl_factor:.3f}'  # Add generator info
                        results.append(row)
                        # print(row)

                # Save results to JSON file
                with open(output_file, 'w') as outfile:
                    json.dump(results, outfile, indent=4)
                print(f"Results saved to {output_file}")


def main():
    args = parse_args()

    # Load YAML configuration
    llm_dicts = load_config(args.llms)
    llm_gen_dict = load_config(args.llm_gen)
    llm_gen_config = LLMGenerateConfig(**llm_gen_dict)

    # Load dataset
    # eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    eval_set = datasets.load_dataset("/root/.cache/huggingface/datasets/tatsu-lab___alpaca_eval")['test']

    model_name = args.target_llm
    model_value = llm_dicts[model_name]

    ctrl_factors = np.linspace(*model_value["ctrl_factor"], args.ctrl_split) * args.factor
    print(ctrl_factors)
    ctrl_factors = list(ctrl_factors)
    ctrl_factors.insert(0, 0.00)
    # ctrl_factors = ctrl_factors[::-1][::2]
    # ctrl_factors = [0.]
    topk = model_value.get('topk', None)
    selector = model_value.get('selector', None)

    wait_for_gpu_memory(args.device, threshold=.3, check_interval=5)

    # Create output directory
    if topk is None:
        output_base_dir = os.path.join(args.output_dir, model_name.replace('/', '_'))
    else:
        output_base_dir = os.path.join(args.output_dir, '{topk}', model_name.replace('/', '_'))

    os.makedirs(output_base_dir, exist_ok=True)

    # Submit the model inference to the executor
    run_model_inference(
        model_name=model_name,
        model_value=model_value,
        eval_set=eval_set,
        llm_gen_config=llm_gen_config,
        # ctrl_factors=ctrl_factors,
        ctrl_factors=ctrl_factors[args.split::args.total],
        # ctrl_factors=reversed(ctrl_factors),
        topk=topk,
        selector=selector,
        batch_size=args.batch_size,
        output_base_dir=output_base_dir,
        device_id=args.device
    )


if __name__ == '__main__':
    main()
