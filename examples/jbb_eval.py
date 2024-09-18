import os
import json
from copy import deepcopy
import yaml
import argparse
import glob

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from jailbreakpipe.pipelines.inference import InferPipeline, InferPipelineConfig
from jailbreakpipe.utils import parse_configs_from_dict

# Load the YAML configuration file


def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

# python jbb_eval.py --input-dir ../results/jbb/repe --output-dir ../results/jbb/repe_judged
# python jbb_eval.py --input-dir ../results/jbb/1 --output-dir ../results/jbb/1_judged


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument('--config', type=str, default="../configs/judge.yaml", help='Path to YAML configuration file')
    parser.add_argument('--input-dir', type=str, default="../results/jbb/repe", help='Input directory containing JSON files')
    parser.add_argument('--output-dir', type=str, default="../results/jbb/repe_judged", help='Output directory')
    parser.add_argument('--repeats', type=int, default=1, help='Number of repeats')
    # parser.add_argument('--judges', type=str, help='Judges configuration file')
    parser.add_argument('--num-workers', type=int, default=8, help='Target LLM name or configuration file')
    return parser.parse_args()


def fill_llms_configs(d, parent_key='', llm_configs=None):
    for key, value in d.items():
        full_key = f'{parent_key}.{key}' if parent_key else key
        if isinstance(value, dict):
            fill_llms_configs(value, full_key, llm_configs)
        else:
            if key.endswith("llm_config") and not value:
                d[key] = llm_configs


def override_config(config_dict, args):
    # Override parameters from command line arguments

    # if args.judges:
    #     with open(args.judges, 'r') as file:
    #         judge_configs = yaml.safe_load(file)
    #         config_dict["judges"] = judge_configs

    fill_llms_configs(config_dict, "", llm_configs=config_dict["defender"]["target_llm_config"])

    if "output_file" not in config_dict["misc"] or config_dict["misc"]["output_file"] is None:
        config_dict["misc"]["output_file"] = os.path.join(
            args.output_dir,
            str(args.repeats),
            config_dict["defender"]["target_llm_config"]["model_name"].replace('/', '_'),
            f'{config_dict["attacker"]["attacker_cls"]}_{config_dict["attacker"]["attacker_name"]}_{config_dict["defender"]["defender_cls"]}.json'
        )
    config_dict["misc"]["repeats"] = args.repeats

    return config_dict


def run_inference(pipe, messages, goal):
    result = pipe.parallel_judging(messages, goal)
    pipe.reset()
    return result


def process_file(json_file, args, attacker_config, defender_config, judge_configs, config_dict):
    # Construct output file path
    output_file = os.path.join(args.output_dir, os.path.relpath(json_file, args.input_dir))

    # Check if output file exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists, skipping.")
        return

    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Initialize pipeline
    pipe = InferPipeline(
        InferPipelineConfig(
            attacker_config=attacker_config,
            defender_config=defender_config,
            judge_configs=judge_configs  # Can be 0, 1, or multiple judges
        ),
        verbose=False
    )

    if 'results' in data:
        data = data['results']

    # Process data
    for i, item in enumerate(tqdm(data, desc=os.path.basename(json_file))):
        # print(item)
        # exit(0)
        messages = deepcopy(item['data'][0]['messages'])
        goal = item['summary']['Goal']
        result = run_inference(pipe, messages, goal)
        item['summary'] = {**item['summary'], **result}

    # Save modified data to the output directory with the same relative file path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "config": config_dict,
            "results": data
        }, f, indent=4)


if __name__ == '__main__':

    args = parse_args()

    # Load and possibly override the YAML configuration
    config_dict = load_config(args.config)
    config_dict = override_config(config_dict, args)

    # Convert YAML dictionary into attacker, defender, and judge configurations
    attacker_config, defender_config, judge_configs = parse_configs_from_dict(config_dict)

    # print(config_dict)
    # Recursively process all JSON files in the input directory, including subdirectories
    json_files = glob.glob(os.path.join(args.input_dir, '**', '*.json'), recursive=True)

    # print(json_files)
    # exit(0)

    # Use ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for json_file in json_files:
            future = executor.submit(process_file, json_file, args, attacker_config, defender_config, judge_configs, config_dict)
            futures.append(future)

        # Optional: Display progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            pass  # You can handle exceptions here if needed

    print(f"Results saved to {args.output_dir}")
