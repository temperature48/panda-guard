import os
import json
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


# Load the YAML configuration file
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def get_gpu_memory_usage(device: str):
    """
    获取指定 GPU 设备的显存使用情况。

    参数:
    - device: 例如 'cuda:0'

    返回:
    - total_mem: 总显存 (MB)
    - used_mem: 已用显存 (MB)
    - free_mem: 空余显存 (MB)
    """
    gpu_id = device.split(':')[-1]  # 获取 GPU ID
    # 使用 nvidia-smi 获取显存信息
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader',
         '--id=' + gpu_id],
        stdout=subprocess.PIPE,
        encoding='utf-8'
    )

    # 解析 nvidia-smi 返回的显存信息
    output = result.stdout.strip()
    total_mem, used_mem, free_mem = map(int, output.split(', '))
    return total_mem, used_mem, free_mem


def wait_for_gpu_memory(device: str, threshold: float = 0.8, check_interval: int = 5):
    """
    阻塞程序直到指定 GPU 的空余显存超过阈值。

    参数:
    - device: GPU 设备（如 'cuda:0'）
    - threshold: 空余显存的阈值（默认 80%，即 0.8）
    - check_interval: 检查间隔时间，单位为秒（默认 5 秒）
    """
    while True:
        total_mem, used_mem, free_mem = get_gpu_memory_usage(device)
        free_ratio = free_mem / total_mem

        print(
            f"GPU {device}: Total: {total_mem}MB, Used: {used_mem}MB, Free: {free_mem}MB ({free_ratio * 100:.2f}% free)")

        if free_ratio >= threshold:
            print(f"GPU {device} has more than {threshold * 100}% free memory. Proceeding...")
            break

        print(f"Waiting for GPU {device} to have more than {threshold * 100}% free memory...")
        time.sleep(check_interval)



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
    return parser.parse_args()


# Parallel inference function for each model
def run_model_inference(model_name, model_value, eval_set, llm_gen_config, ctrl_factors, batch_size, output_base_dir, device_id):
    device_map = device_id  # Assign each model to a specific GPU
    # if model_value['device_map'] is None:
    model_value['device_map'] = device_map  # Update device_map in the model config
    model_value['ctrl_factor'] = 0.0  # Set the control factor to 0.0

    llm_config = RepeLLMConfig(**model_value)
    print(llm_config)
    llm = create_llm(llm_config)

    # For each control factor
    for ctrl_factor in ctrl_factors:
        results = []

        output_file = os.path.join(output_base_dir, f'full_{ctrl_factor:.3f}.json')
        print(output_file)
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping...")
            continue  # Skip if file already exists

        # Set the LLM activation according to the control factor
        llm.set_activations(ctrl_factor)

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

    ctrl_factors = np.linspace(*model_value["ctrl_factor"], args.ctrl_split)
    ctrl_factors = list(ctrl_factors)
    ctrl_factors.insert(0, 0.00)
    # ctrl_factors = ctrl_factors[::-1][::2]
    # ctrl_factors = [0.]

    # wait_for_gpu_memory(args.device, threshold=.875, check_interval=5)

    # Create output directory
    output_base_dir = os.path.join(args.output_dir, model_name.replace('/', '_'))
    os.makedirs(output_base_dir, exist_ok=True)

    # Submit the model inference to the executor
    run_model_inference(
        model_name=model_name,
        model_value=model_value,
        eval_set=eval_set,
        llm_gen_config=llm_gen_config,
        ctrl_factors=ctrl_factors,
        # ctrl_factors=ctrl_factors[args.split::8],
        # ctrl_factors=reversed(ctrl_factors),
        batch_size=args.batch_size,
        output_base_dir=output_base_dir,
        device_id=args.device
    )


if __name__ == '__main__':
    main()
