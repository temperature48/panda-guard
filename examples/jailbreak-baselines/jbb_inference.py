# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/16 19:46
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : run_inference.py
# explain   : Run inference using a specific attack and defense configuration.

import os
import json
from typing import Dict, Any

import pandas as pd
import yaml
import argparse

from tqdm import tqdm

from jailbreakpipe.llms import create_llm, HuggingFaceLLMConfig
from jailbreakpipe.pipelines.inference import InferPipeline, InferPipelineConfig
from jailbreakpipe.utils import parse_configs_from_dict


def load_yaml(yaml_file) -> Dict[str, Any]:
    """Load a YAML file."""
    if yaml_file is None or not os.path.exists(yaml_file):
        return {}
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def parse_args():
    """Parse command line arguments for running inference."""
    parser = argparse.ArgumentParser(description="Run inference with a specific attack and defense configuration.")
    parser.add_argument('--config', type=str, default='../../configs/tasks/jbb.yaml', help='Path to the main configuration file')
    parser.add_argument('--attack', type=str, required=False, help='Path to the attack configuration file')
    parser.add_argument('--defense', type=str, required=False, help='Path to the defense configuration file')
    parser.add_argument('--llm', type=str, required=False, help='Path to the target LLM file')
    parser.add_argument('--llm-gen', type=str, default='../../config/defenses/llm_gen/jbb.yaml', help='Path to the LLM generator file')
    parser.add_argument('--device', type=str, default=None, help='Device to run the model on')
    parser.add_argument('--output-dir', type=str, default="../../benchmarks/jbb", help='Output directory')
    parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat the experiment')
    return parser.parse_args()


def run_inference(args):
    """Run inference using the provided configurations."""
    # Load the main configuration and override attack/defense paths.
    config_dict = load_yaml(args.config)
    print(args.config)
    config_dict["attacker"] = load_yaml(args.attack)
    config_dict["defender"] = load_yaml(args.defense)

    print(config_dict)

    llm_dict = load_yaml(args.llm)
    llm_gen_dict = load_yaml(args.llm_gen)

    if llm_dict:
        config_dict["defender"]["target_llm_config"] = llm_dict
    else:
        llm_dict = config_dict["defender"]["target_llm_config"]

    if args.device:
        config_dict["defender"]["target_llm_config"]["device_map"] = args.device

    if llm_gen_dict:
        config_dict["defender"]["target_llm_gen_config"].update(llm_gen_dict)

    # Load and parse attacker and defender configurations.
    attacker_config, defender_config, _ = parse_configs_from_dict(config_dict)

    # Load CSV input file.
    df = pd.read_csv(config_dict["misc"]["input_file"])

    # Update output file path.
    output_file = os.path.join(
        args.output_dir,
        llm_dict['model_name'].replace('/', '_'),
        f'{config_dict["attacker"]["attacker_cls"]}_{config_dict["attacker"]["attacker_name"]}',
        f'{config_dict["defender"]["defender_cls"]}',
        'results.json'
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    yaml.dump(config_dict, open(output_file.replace('results.json', 'config.yaml'), 'w'))

    # Initialize the pipeline.
    pipe = InferPipeline(
        InferPipelineConfig(
            attacker_config=attacker_config,
            defender_config=defender_config
        ),
        verbose=False
    )

    # Run inference for each row.
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        messages = [{"role": "user", "content": row["Goal"]}]
        result = [pipe(messages, row["Goal"], request_reformulated=row.get(attacker_config.attacker_name, None)) for _ in range(args.repeats)]
        results.append({
            "goal": row["Goal"],
            "data": result
        })
        pipe.reset()
        # print(results[-1])

    # Save results to the output file.
    with open(output_file, 'w') as outfile:
        json.dump({"config": config_dict, "results": results}, outfile, indent=4)
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    args = parse_args()
    run_inference(args)
