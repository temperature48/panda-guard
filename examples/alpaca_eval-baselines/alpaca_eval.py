# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/10/12 15:21
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : alpaca_eval.py
# explain   :
import logging
import os
import subprocess
import argparse
import json
from concurrent.futures import ThreadPoolExecutor


def execute_command(model_output, reference_output, output_directory):
    if os.path.exists(os.path.join(output_directory, "alpaca_eval_llama3_70b_fn/annotations.json")):
        logging.warning(f"Skipping {model_output} as {output_directory} already exists.")
        return None

    if not validate_json(model_output):
        logging.warning(f"Skipping invalid JSON file: {model_output}")
        return None

    logging.info(f"Processing {model_output} and {reference_output} to {output_directory}")

    command = [
        "HF_ENDPOINT=https://hf-mirror.com",
        "alpaca_eval",
        "--model_outputs",
        model_output,
        "--reference_outputs",
        reference_output,
        "--output_path",
        output_directory,
        "--annotators_config",
        args.evaluator,
        # "alpaca_eval_llama3_70b_fn_copy"
    ]
    logging.info(f"Executing: {' '.join(command)}")
    try:
        result = subprocess.run(' '.join(command), shell=True, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with error: {e}")
        return e.returncode


def validate_json(json_file):
    try:
        with open(json_file, 'r') as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        logging.error(f"Error in {json_file}: {e}")
        return False


def process_directory(input_root, output_root, threads):
    # 遍历目录，查找包含 NoneDefender.json 的路径，并进行评测
    for root, dirs, files in os.walk(input_root):
        reference_output = None
        model_outputs = []

        for d in dirs:
            file = os.path.join(d, 'results.json')
            if file == 'NoneDefender/results.json' or '0.000' in file:
                reference_output = os.path.join(root, file)
            elif file.endswith(".json") and file != 'NoneDefender/results.json':
                model_outputs.append(os.path.join(root, file))

        if model_outputs and not reference_output:
            logging.warning(f"Reference output not found in {root}.")

        if reference_output:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                # for model_output in model_outputs[3::5]:
                for model_output in model_outputs:

                    logging.debug(model_output)
                    relative_path = os.path.relpath(root, input_root)
                    output_directory = os.path.join(output_root, relative_path, model_output.split('/')[-2])

                    # 创建输出目录
                    os.makedirs(output_directory, exist_ok=True)

                    # 使用线程池并行执行
                    executor.submit(execute_command, model_output, reference_output, output_directory)

    logging.info("All commands completed.")


# python alpaca_eval_leaderboard.py ../results/alpaca_eval/repe_.9 ../results/alpaca_eval_leaderboard/repe_.9  --threads 4
# python alpaca_eval_leaderboard.py ../results/alpaca_eval/1 ../results/alpaca_eval_leaderboard/1  --threads 4
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute alpaca_eval on JSON files in a directory.")
    parser.add_argument('--input-dir', type=str, default="../../benchmarks/alpaca_eval/", help='Input directory containing JSON files or YAML file containing list of files')
    parser.add_argument('--output-dir', type=str, default="../../benchmarks/alpaca_eval_judged/", help='Output directory')
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use.")
    parser.add_argument("--evaluator", type=str, default='alpaca_eval_llama3_70b_fn', help="Number of threads to use.")
    parser.add_argument(
        '--log-level',
        type=str,
        default="WARNING",
        help="Logging level, e.g., DEBUG, INFO, WARNING, ERROR"
    )

    args = parser.parse_args()
    logging.basicConfig(level=eval(f'logging.{args.log_level}'), format='%(asctime)s - %(levelname)s - %(message)s')
    # 执行目录处理
    process_directory(args.input_dir, args.output_dir, args.threads)

