# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/10/10 21:56
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : run_all_inference.py.py
# explain   :

import os
import subprocess
import argparse
from tqdm import tqdm


def parse_args():
    """Parse command line arguments for running all experiments."""
    parser = argparse.ArgumentParser(description="Run all experiments for different attack and defense configurations.")
    parser.add_argument('--config', type=str, required=True, help='Path to the main configuration file')
    parser.add_argument('--llms', type=str, required=True, help='Path to list of available LLMs')
    parser.add_argument('--target-llm', type=str, required=True, help='Target LLM to use')
    parser.add_argument('--attack-dir', type=str, required=True, help='Path to attack configurations directory')
    parser.add_argument('--defense-dir', type=str, required=True, help='Path to defense configurations directory')
    parser.add_argument('--output-dir', type=str, default="./results", help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat the experiment')
    return parser.parse_args()


def run_all_experiments(args):
    """Iterate over attack and defense configurations and run inference for each combination."""
    # List attack and defense configuration files.
    attack_files = [os.path.join(args.attack_dir, f) for f in os.listdir(args.attack_dir) if f.endswith('.yaml')]
    defense_files = [os.path.join(args.defense_dir, f) for f in os.listdir(args.defense_dir) if f.endswith('.yaml')]

    # Iterate over each combination of attack and defense configurations.
    for attack_file in tqdm(attack_files, desc="Attacks"):
        for defense_file in tqdm(defense_files, desc=f"Defenses for {attack_file}"):
            # Build the command to run `run_inference.py`.
            command = [
                "python", "run_inference.py",
                "--config", args.config,
                "--llms", args.llms,
                "--target-llm", args.target_llm,
                "--attack", attack_file,
                "--defense", defense_file,
                "--device", args.device,
                "--output-dir", args.output_dir,
                "--repeats", str(args.repeats)
            ]
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command, check=True)


if __name__ == '__main__':
    args = parse_args()
    run_all_experiments(args)
