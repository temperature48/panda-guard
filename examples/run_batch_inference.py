# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/5 14:39
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : run_batch_inference.py
# explain   :

import os
import subprocess
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed


# def parse_args():
parser = argparse.ArgumentParser(description="Run batch inference for multiple configurations")
parser.add_argument('--inference-script', type=str, default="jbb_inference", help='Path to the jbb_inference.py script')
parser.add_argument('--attack-dir', type=str, default="../configs/attacks/transfer", help='Path to attack configurations directory')
parser.add_argument('--defense-dir', type=str, default="../configs/defenses/", help='Path to defense configurations directory')
parser.add_argument('--llms', type=str, default="../configs/llms.yaml", help='Path to llms.yaml configuration file')
parser.add_argument('--max-workers', type=int, default=16, help='Number of parallel threads to use')
parser.add_argument('--output-dir', type=str, default="../results/", help='Output directory for batch results')
parser.add_argument('--repeats', type=int, default=10, help='Number of times to repeat the experiment')
args = parser.parse_args()

args.output_dir = os.path.join(args.output_dir, "_".join(args.inference_script.split('_')[:-1]))


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def run_inference_for_llm(attack_file, defense_file, llm_name, llms_path, output_dir, inference_script):
    # Construct the output directory based on attack, defense, and llm names
    # Construct the command to run the inference script with the current configuration
    command = [
        'python', inference_script + '.py',
        '--config', '../configs/default.yaml',  # Assuming default config is needed
        '--attack', attack_file,
        '--defense', defense_file,
        '--llms', llms_path,
        '--target-llm', llm_name,
        '--output-dir', output_dir,
    ]
    # print(command)
    # Run the inference script
    result = subprocess.run(command, capture_output=True, text=True)
    return command, result.stdout, result.stderr


def main():
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the LLM configurations from llms.yaml
    llm_configs = load_yaml(args.llms)
    # List all attack and defense configurations
    attack_files = [os.path.join(args.attack_dir, f) for f in os.listdir(args.attack_dir) if f.endswith('.yaml')]
    # attack_files = ["../configs/attacks/transfer/original.yaml"]
    defense_files = [os.path.join(args.defense_dir, f) for f in os.listdir(args.defense_dir) if f.endswith('.yaml')]

    # Using ThreadPoolExecutor for parallel LLM evaluations
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for attack_file in attack_files:
            for defense_file in defense_files:
                for llm_name, llm_config in llm_configs.items():
                    # Submit the inference task to the thread pool
                    futures.append(executor.submit(
                        run_inference_for_llm,
                        attack_file,
                        defense_file,
                        llm_name,
                        args.llms,
                        args.output_dir,
                        args.inference_script,
                    ))

        # Process results as they complete
        for future in as_completed(futures):
            command, stdout, stderr = future.result()
            if stderr and "error" in stderr.lower():
                print(f"Error occurred during inference, Command: {command}, error: {stderr}.")
            else:
                print(stdout, end='')


if __name__ == '__main__':
    main()
