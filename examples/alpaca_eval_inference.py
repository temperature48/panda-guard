# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/5 23:09
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : alpaca_eval_inference.py
# explain   :

# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/4 17:58
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : eval_jbb.py
# explain   :

import os
import json
import pandas as pd
import yaml
import argparse

import datasets

from concurrent.futures import ThreadPoolExecutor
from jailbreakpipe.pipelines.inference import InferPipeline, InferPipelineConfig
from jailbreakpipe.utils import parse_configs_from_dict


# Load the YAML configuration file
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument('--config', type=str, default="../configs/alpaca_eval.yaml", help='Path to YAML configuration file')
    parser.add_argument('--llms', type=str, default="../configs/llms.yaml", help='Path to list of available LLMs')
    parser.add_argument('--attack', type=str, default=None)
    parser.add_argument('--defense', type=str, default=None)
    parser.add_argument('--judges', type=str, default=None)
    parser.add_argument('--target-llm', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default="../results/alpaca_eval", help='Output directory')
    parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat the experiment')
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
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": None}
    ]
    result = pipe(messages, None, request_reformulated=row["instruction"])
    pipe.reset()
    print(result)
    return result


def main():
    args = parse_args()

    # Load and possibly override the YAML configuration
    config_dict = load_config(args.config)
    config_dict = override_config(config_dict, args)

    if os.path.exists(config_dict["misc"]["output_file"]):
        print(f"File {config_dict['misc']['output_file']} already exists. Skipping...")
        return

    # Load CSV input file
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

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

    # Parallel processing with ThreadPoolExecutor
    results = []
    for i, row in enumerate(eval_set):
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


if __name__ == '__main__':
    main()
