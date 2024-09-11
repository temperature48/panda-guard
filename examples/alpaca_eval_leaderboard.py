import os
import subprocess
import argparse
import json
from concurrent.futures import ThreadPoolExecutor


def execute_command(model_output, reference_output, output_directory):
    if os.path.exists(os.path.join(output_directory, "weighted_alpaca_eval_qwen_2/annotations.json")):
        print(f"Skipping {model_output} as {output_directory} already exists.")
        return None

    if not validate_json(model_output):
        print(f"Skipping invalid JSON file: {model_output}")
        return None

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
        "weighted_alpaca_eval_qwen_2"
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
            if file == 'NoneDefender.json':
                reference_output = os.path.join(root, file)
            elif file.endswith(".json") and file != 'NoneDefender.json':
                model_outputs.append(os.path.join(root, file))

        if reference_output:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                for model_output in model_outputs:
                    relative_path = os.path.relpath(root, input_root)
                    output_directory = os.path.join(output_root, relative_path, os.path.splitext(os.path.basename(model_output))[0])

                    # 创建输出目录
                    os.makedirs(output_directory, exist_ok=True)

                    # 使用线程池并行执行
                    executor.submit(execute_command, model_output, reference_output, output_directory)

    print("All commands completed.")


# 主函数，解析输入参数
def main():
    parser = argparse.ArgumentParser(description="Execute alpaca_eval on JSON files in a directory.")
    parser.add_argument("input_root", type=str, help="The root directory to search for JSON files.")
    parser.add_argument("output_root", type=str, help="The root directory to save the evaluation results.")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use.")

    args = parser.parse_args()

    # 执行目录处理
    process_directory(args.input_root, args.output_root, args.threads)


if __name__ == "__main__":
    main()
