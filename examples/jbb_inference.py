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
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    parser.add_argument('--target-llm', type=str, default=None)
    parser.add_argument('--attack-cls', type=str, default=None)
    parser.add_argument('--attack-name', type=str, default=None)
    parser.add_argument('--defense-cls', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default="../results", help='Output directory')
    return parser.parse_args()


def override_config(config_dict, args):
    # Override parameters from command line arguments
    if args.target_llm:
        config_dict["defender"]["target_llm_config"]["model_name"] = args.target_llm

    if args.attack_cls:
        config_dict["attacker"]["attacker_cls"] = args.attack_cls
    if args.attack_name:
        config_dict["attacker"]["attacker_name"] = args.attacker_name

    if args.defense_cls:
        config_dict["defender"]["defender_cls"] = args.defense_cls

    if config_dict["misc"]["output_file"] is None:
        config_dict["misc"]["output_file"] = os.path.join(
            args.output_dir,
            config_dict["defender"]["target_llm_config"]["model_name"],
            f'{config_dict["attacker"]["attacker_cls"]}_{config_dict["attacker"]["attacker_name"]}_{config_dict["defender"]["defender_cls"]}.json'
        )

    return config_dict


def run_inference(pipe, row, attacker_config):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": row["Goal"]}
    ]
    result = pipe(messages, row["Goal"], request_reformulated=row.get(attacker_config.attacker_name, None))
    pipe.reset()
    return result


def main():
    args = parse_args()

    # Load and possibly override the YAML configuration
    config_dict = load_config(args.config)
    config_dict = override_config(config_dict, args)

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
    for i, row in df[:3].iterrows():
        result = run_inference(pipe, row, attacker_config)
        print(result)
        results.append(result)

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


if __name__ == '__main__':
    main()
