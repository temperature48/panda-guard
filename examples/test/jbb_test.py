# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/16 19:46
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : jbb_gen.py
# explain   :

import os
import json
import pandas as pd
import yaml
import argparse

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from jailbreakpipe.llms import create_llm, HuggingFaceLLMConfig
from jailbreakpipe.pipelines.inference import InferPipeline, InferPipelineConfig
from jailbreakpipe.utils import parse_configs_from_dict


# Load the YAML configuration file
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument('--config', type=str, default="../../configs/default.yaml", help='Path to YAML configuration file')
    parser.add_argument('--llms', type=str, default="../../configs/llms.yaml", help='Path to list of available LLMs')
    parser.add_argument('--target-llm', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    parser.add_argument('--attack', type=str)
    parser.add_argument('--defense', type=str)
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
    if args.attack:
        with open(args.attack, 'r') as file:
            attack_config = yaml.safe_load(file)
            config_dict["attacker"] = attack_config

    if args.defense:
        with open(args.defense, 'r') as file:
            defense_config = yaml.safe_load(file)
            defense_config['target_llm_config']["model_name"] = args.target_llm
            config_dict["defender"] = defense_config

    fill_llms_configs(config_dict, "", llm_configs=config_dict["defender"]["target_llm_config"])

    print(config_dict)

    return config_dict


def run_inference(pipe, row, attacker_config):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": row["Goal"]}
    ]
    if 'gemma' in args.target_llm:
        messages = [{"role": "user", "content": row["Goal"]}]
    result = pipe(messages, row["Goal"], request_reformulated=row.get(attacker_config.attacker_name, None))
    pipe.reset()
    return result


if __name__ == '__main__':
    args = parse_args()

    llm_config = HuggingFaceLLMConfig(
        model_name=args.target_llm,
        device_map=args.device,
    )

    config_dict = load_config(args.config)
    config_dict = override_config(config_dict, args)

    df = pd.read_csv(config_dict["misc"]["input_file"])

    attacker_config, defender_config, judge_configs = parse_configs_from_dict(config_dict)

    # Initialize the pipeline
    pipe = InferPipeline(
        InferPipelineConfig(
            attacker_config=attacker_config,
            defender_config=defender_config,
            judge_configs=judge_configs  # Can be 0, 1, or multiple judges
        ),
        verbose=False
    )

    results = []
    for i, row in df.iterrows():
        result = []
        judge_kwargs = {}

        result.append(run_inference(pipe, row, attacker_config))
        for key, value in result[-1]["judges"].items():
            if key not in judge_kwargs:
                judge_kwargs[key] = 1
            judge_kwargs[key] = max(judge_kwargs[key], value)

        judge_kwargs["Goal"] = row["Goal"]
        results.append({
            "summary": judge_kwargs,
            "data": result,
        })
        print(results[-1])

