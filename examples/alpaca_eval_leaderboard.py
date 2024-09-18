import os
import subprocess
import argparse
import json
from concurrent.futures import ThreadPoolExecutor


def execute_command(model_output, reference_output, output_directory):
    if os.path.exists(os.path.join(output_directory, "alpaca_eval_llama3_70b_fn/annotations.json")):
        print(f"Skipping {model_output} as {output_directory} already exists.")
        return None

    if not validate_json(model_output):
        print(f"Skipping invalid JSON file: {model_output}")
        return None

    print(f"Processing {model_output} and {reference_output} to {output_directory}")

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
    print(f"Executing: {' '.join(command)}")
    try:
        result = subprocess.run(' '.join(command), shell=True, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        return e.returncode


def validate_json(json_file):
    try:
        with open(json_file, 'r') as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        print(f"Error in {json_file}: {e}")
        return False


def process_directory(input_root, output_root, threads):
    # 遍历目录，查找包含 NoneDefender.json 的路径，并进行评测
    for root, dirs, files in os.walk(input_root):
        reference_output = None
        model_outputs = []

        for file in files:
            if file == 'NoneDefender.json' or '0.000' in file:
                reference_output = os.path.join(root, file)
            elif file.endswith(".json") and file != 'NoneDefender.json':
                model_outputs.append(os.path.join(root, file))

        if model_outputs and not reference_output:
            reference_output = os.path.join(root.replace('repe', '1'), 'NoneDefender.json')

        if reference_output:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                for model_output in model_outputs[3::5]:
                    print(model_output)
                    relative_path = os.path.relpath(root, input_root)
                    output_directory = os.path.join(output_root, relative_path, os.path.splitext(os.path.basename(model_output))[0])

                    # 创建输出目录
                    os.makedirs(output_directory, exist_ok=True)

                    # 使用线程池并行执行
                    executor.submit(execute_command, model_output, reference_output, output_directory)

    print("All commands completed.")


# python alpaca_eval_leaderboard.py ../results/alpaca_eval/repe_.9 ../results/alpaca_eval_leaderboard/repe_.9  --threads 4
# python alpaca_eval_leaderboard.py ../results/alpaca_eval/1 ../results/alpaca_eval_leaderboard/1  --threads 4
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute alpaca_eval on JSON files in a directory.")
    parser.add_argument("input_root", type=str, help="The root directory to search for JSON files.")
    parser.add_argument("output_root", type=str, help="The root directory to save the evaluation results.")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use.")
    parser.add_argument("--evaluator", type=str, default='alpaca_eval_llama3_70b_fn', help="Number of threads to use.")

    args = parser.parse_args()

    # 执行目录处理
    process_directory(args.input_root, args.output_root, args.threads)

