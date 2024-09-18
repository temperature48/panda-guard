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

from concurrent.futures import ThreadPoolExecutor
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
    parser.add_argument('--target-llm', type=str, default=None)
    parser.add_argument('--attack', type=str, default=None)
    parser.add_argument('--defense', type=str, default=None)
    parser.add_argument('--judges', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default="../../results/jbb", help='Output directory')
    parser.add_argument('--repeats', type=int, default=10, help='Number of times to repeat the experiment')
    return parser.parse_args()


args = parse_args()


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
            config_dict["defender"] = defense_config

    if args.judges:
        with open(args.judges, 'r') as file:
            judge_configs = yaml.safe_load(file)
            config_dict["judges"] = judge_configs

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

    # print(config_dict)

    # print(config_dict)

    if config_dict["misc"]["output_file"] is None:
        config_dict["misc"]["output_file"] = os.path.join(
            args.output_dir,
            str(args.repeats),
            config_dict["defender"]["target_llm_config"]["model_name"].replace('/', '_'),
            f'{config_dict["attacker"]["attacker_cls"]}_{config_dict["attacker"]["attacker_name"]}_{config_dict["defender"]["defender_cls"]}.json'
        )
    config_dict["misc"]["repeats"] = args.repeats

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


def main():

    # Load and possibly override the YAML configuration
    config_dict = load_config(args.config)
    config_dict = override_config(config_dict, args)

    if os.path.exists(config_dict["misc"]["output_file"]):
        print(f"File {config_dict['misc']['output_file']} already exists. Skipping...")
        return

    # Load CSV input file
    df = pd.read_csv(config_dict["misc"]["input_file"])

    # Convert YAML dictionary into attacker, defender, and judge configurations
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

    # Parallel processing with ThreadPoolExecutor
    results = []
    for i, row in df.iterrows():
        result = []
        judge_kwargs = {}
        for _ in range(args.repeats):
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

    os.makedirs('/'.join(config_dict["misc"]["output_file"].split('/')[:-1]), exist_ok=True)
    # Save results to JSON file
    with open(config_dict["misc"]["output_file"], 'w') as outfile:
        json.dump(
            {
                "config": config_dict,
                "results": results
            },
            outfile,
            indent=4
        )
    print(f"Results saved to {config_dict['misc']['output_file']}")


if __name__ == '__main__':
    main()
