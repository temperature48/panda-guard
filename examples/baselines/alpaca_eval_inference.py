# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/16 20:37
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : alpaca_eval_inference.py
# explain   :
# encoding: utf-8


import os
import json
import pandas as pd
import yaml
import argparse

import datasets

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from jailbreakpipe.llms import HuggingFaceLLMConfig, create_llm
from jailbreakpipe.pipelines.inference import InferPipeline, InferPipelineConfig
from jailbreakpipe.utils import parse_configs_from_dict, wait_for_gpu_memory


# Load the YAML configuration file
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument('--config', type=str, default="../../configs/alpaca_eval.yaml", help='Path to YAML configuration file')
    parser.add_argument('--llms', type=str, default="../../configs/llms.yaml", help='Path to list of available LLMs')
    parser.add_argument('--target-llm', type=str, default=None)
    parser.add_argument('--defense-dir', type=str, default="../../configs/defenses/", help='Path to defense configurations directory')
    parser.add_argument('--output-dir', type=str, default="../../results/alpaca_eval", help='Output directory')
    parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat the experiment')
    parser.add_argument('--device', type=str, default='cuda:0', help='Number of GPUs to use')
    parser.add_argument('--part', type=int, default=0)
    return parser.parse_args()



# @lru_cache(maxsize=None)
def fill_llms_configs(d, parent_key='', llm_configs=None):
    for key, value in d.items():
        full_key = f'{parent_key}.{key}' if parent_key else key
        if isinstance(value, dict):
            fill_llms_configs(value, full_key, llm_configs)  # 递归调用
        else:
            if key.endswith("llm_config") and not value:
                d[key] = llm_configs


def override_config(config_dict, args):
    # Override parameters from command line arguments

    if args.defense:
        with open(args.defense, 'r') as file:
            defense_config = yaml.safe_load(file)
            config_dict["defender"] = defense_config

    if args.target_llm:
        # if target_llm is a Yaml file, load it, else find the model_name in llms.yaml
        if args.target_llm.endswith('.yaml'):
            with open(args.target_llm, 'r') as file:
                config_dict["defender"]["target_llm_config"] = yaml.safe_load(file)
        else:
            with open(args.llms, 'r') as file:
                llm_configs = yaml.safe_load(file)
                config_dict["defender"]["target_llm_config"] = {
                    "model_name": args.target_llm,
                    **llm_configs[args.target_llm]
                }

    fill_llms_configs(config_dict, "", llm_configs=config_dict["defender"]["target_llm_config"])

    if config_dict["misc"]["output_file"] is None:
        config_dict["misc"]["output_file"] = os.path.join(
            args.output_dir,
            str(args.repeats),
            config_dict["defender"]["target_llm_config"]["model_name"].replace('/', '_'),
            f'{config_dict["defender"]["defender_cls"]}.json'
        )
    config_dict["misc"]["repeats"] = args.repeats

    return config_dict


def run_inference(pipe, row, attacker_config):
    if 'gemma' not in args.target_llm:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": None}
        ]
    else:
        messages = [{"role": "user", "content": None}]
    result = pipe(messages, None, request_reformulated=row["instruction"])
    pipe.reset()
    # print(result)
    return result


# python alpaca_eval_inference.py --target-llm google/gemma-2-2b-it --device cuda:4 --part 0
# python alpaca_eval_inference.py --target-llm google/gemma-2-2b-it --device cuda:5 --part 1
# python alpaca_eval_inference.py --target-llm google/gemma-2-2b-it --device cuda:6 --part 2
# python alpaca_eval_inference.py --target-llm google/gemma-2-2b-it --device cuda:7 --part 3

# python alpaca_eval_inference.py --target-llm meta-llama/Meta-Llama-3-8B-Instruct --device cuda:1 --part 0
# python alpaca_eval_inference.py --target-llm meta-llama/Meta-Llama-3-8B-Instruct --device cuda:4 --part 1

# python alpaca_eval_inference.py --target-llm mistralai/Mistral-7B-Instruct-v0.3 --device cuda:2 --part 0
# python alpaca_eval_inference.py --target-llm mistralai/Mistral-7B-Instruct-v0.3 --device cuda:3 --part 1

# python alpaca_eval_inference.py --target-llm microsoft/Phi-3-mini-4k-instruct --device cuda:3 --part 0
# python alpaca_eval_inference.py --target-llm microsoft/Phi-3-mini-4k-instruct --device cuda:2 --part 1

# python alpaca_eval_inference.py --target-llm meta-llama/Meta-Llama-3.1-8B-Instruct --device cuda:4 --part 0
# python alpaca_eval_inference.py --target-llm meta-llama/Meta-Llama-3.1-8B-Instruct --device cuda:1 --part 1

# python alpaca_eval_inference.py --target-llm Qwen/Qwen1.5-7B-Chat --device cuda:5 --part 0
# python alpaca_eval_inference.py --target-llm Qwen/Qwen1.5-7B-Chat --device cuda:0 --part 1

# python alpaca_eval_inference.py --target-llm google/gemma-2-9b-it --device cuda:6 --part 0
# python alpaca_eval_inference.py --target-llm google/gemma-2-9b-it --device cuda:7 --part 1

# CUDA_VISIBLE_DEVICES=0,1 python alpaca_eval_inference.py --target-llm meta-llama/Meta-Llama-3-70B-Instruct --device auto --part 0
# CUDA_VISIBLE_DEVICES=2,3 python alpaca_eval_inference.py --target-llm meta-llama/Meta-Llama-3-70B-Instruct --device auto --part 1
# CUDA_VISIBLE_DEVICES=4,5 python alpaca_eval_inference.py --target-llm meta-llama/Meta-Llama-3-70B-Instruct --device auto --part 2
# CUDA_VISIBLE_DEVICES=6,7 python alpaca_eval_inference.py --target-llm meta-llama/Meta-Llama-3-70B-Instruct --device auto --part 3

if __name__ == '__main__':
    args = parse_args()
    defense_files = [os.path.join(args.defense_dir, f) for f in os.listdir(args.defense_dir) if f.endswith('.yaml')]

    llm_config = HuggingFaceLLMConfig(
        model_name=args.target_llm,
        device_map=args.device,
    )
    wait_for_gpu_memory('cuda:0', threshold=.7, check_interval=5)

    llm = create_llm(llm_config)

    # for defense_file in tqdm(defense_files):
    for defense_file in tqdm(defense_files[args.part::4]):
        args.defense = defense_file
        # Load and possibly override the YAML configuration
        config_dict = load_config(args.config)
        config_dict = override_config(config_dict, args)

        if os.path.exists(config_dict["misc"]["output_file"]):
            print(f"File {config_dict['misc']['output_file']} already exists. Skipping...")
            continue

        print(f"File {config_dict['misc']['output_file']} does not exist. Running...")

        # Load CSV input file
        # eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
        eval_set = datasets.load_dataset("/root/.cache/huggingface/datasets/tatsu-lab___alpaca_eval")['test']

        # Convert YAML dictionary into attacker, defender, and judge configurations
        attacker_config, defender_config, _ = parse_configs_from_dict(config_dict)

        # Initialize the pipeline
        pipe = InferPipeline(
            InferPipelineConfig(
                attacker_config=attacker_config,
                defender_config=defender_config,
                judge_configs=[]  # Can be 0, 1, or multiple judges
            ),
            verbose=False
        )

        pipe.defender.target_llm = llm
        config_dict['defender']['target_llm_config'] = {
            'model_name': args.target_llm,
            'device_map': args.device,
        }

        # Parallel processing with ThreadPoolExecutor
        results = []
        for i, row in tqdm(enumerate(eval_set), desc=f"Running inference {args.defense}"):
            res = run_inference(pipe, row, attacker_config)
            row["output"] = res["messages"][-1]["content"]
            row["generator"] = (
                    config_dict["defender"]["target_llm_config"]["model_name"].replace('/', '_')
                    + '_'
                    + config_dict["defender"]["defender_cls"]
            )
            # print(row)
            results.append(row)

        os.makedirs('/'.join(config_dict["misc"]["output_file"].split('/')[:-1]), exist_ok=True)
        # Save results to JSON file
        with open(config_dict["misc"]["output_file"], 'w') as outfile:
            json.dump(results, outfile, indent=4)
        print(f"Results saved to {config_dict['misc']['output_file']}")

