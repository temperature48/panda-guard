import os
import shutil
import sys
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

import matplotlib
import yaml
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm


ROOT_DIR = "/mnt/home/floyed/panda-guard"
JBB_DIR = os.path.join(ROOT_DIR, "benchmarks", "jbb_judged")
ALPACA_DIR = os.path.join(ROOT_DIR, "benchmarks", "alpaca_eval_judged")
DATA_DIR = os.path.join(ROOT_DIR, "data")
FIGURE_DIR = os.path.join(ROOT_DIR, "examples", "figures")

BENCHMARK_INPUT_DIR = os.path.join(ROOT_DIR, "benchmarks")
BENCHMARK_OUTPUT_DIR = os.path.join(ROOT_DIR, "benchmark_sorted")
os.makedirs(BENCHMARK_OUTPUT_DIR, exist_ok=True)

from tqdm import tqdm

# 需要跳过的路径名称列表
SKIP_PATHS = [
    "new_ica",
    "Llama-3.1-70B-Instruct",
    "Llama-3.3-70B-Instruct",
    "Moonshot-v1",
    "qwen-max-0125",
    "Mistral-Large-2411"
]

# 需要删除的key模式
KEYS_TO_REMOVE = ["api_key", "base_url"]

def process_path_name(path_name):
    """处理路径名称：去掉'new_'前缀和'Defender'后缀"""
    # 去掉'new_'前缀
    if path_name.startswith("new_"):
        path_name = path_name[4:]

    # 去掉'Defender'后缀
    if path_name.endswith("Defender"):
        path_name = path_name[:-8]  # 'Defender'的长度是8

    return path_name

def remove_sensitive_keys(data):
    """递归删除包含敏感关键字的键值对"""
    if isinstance(data, dict):
        # 创建一个包含需要删除的键的列表
        keys_to_delete = []
        for key in data:
            # 检查键名是否包含敏感关键字
            if any(sensitive_key in key.lower() for sensitive_key in KEYS_TO_REMOVE):
                keys_to_delete.append(key)
            # 递归处理嵌套的字典
            elif isinstance(data[key], (dict, list)):
                data[key] = remove_sensitive_keys(data[key])

        # 删除包含敏感关键字的键
        for key in keys_to_delete:
            del data[key]

    elif isinstance(data, list):
        # 递归处理列表中的每个元素
        for i in range(len(data)):
            if i < len(data):  # 确保索引仍然有效
                data[i] = remove_sensitive_keys(data[i])

    return data

def process_json_yaml_file(src_path, dest_path):
    """处理JSON或YAML文件，删除敏感信息"""
    # 检测文件类型并加载内容
    with open(src_path, 'r', encoding='utf-8') as file:
        content = file.read()

        # 尝试加载为JSON
        try:
            if src_path.lower().endswith('.json'):
                data = json.loads(content)
                is_json = True
            elif src_path.lower().endswith(('.yaml', '.yml')):
                data = yaml.safe_load(content)
                is_json = False
            else:
                # 不是JSON或YAML文件，直接复制
                shutil.copy2(src_path, dest_path)
                return
        except Exception as e:
            print(f"警告: 无法解析文件 {src_path} 作为 {'JSON' if src_path.lower().endswith('.json') else 'YAML'}: {e}")
            # 如果解析失败，直接复制原文件
            shutil.copy2(src_path, dest_path)
            return

    # 删除敏感键值对
    cleaned_data = remove_sensitive_keys(data)

    # 确保目标目录存在
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # 将处理后的数据写回文件
    with open(dest_path, 'w', encoding='utf-8') as file:
        if is_json:
            json.dump(cleaned_data, file, indent=2, ensure_ascii=False)
        else:
            yaml.dump(cleaned_data, file, default_flow_style=False, allow_unicode=True)

# 计算源目录中的文件总数（用于进度条）
def count_files(directory, skip_paths=[]):
    """递归计算目录中的文件数量，排除跳过的路径"""
    count = 0
    for root, dirs, files in os.walk(directory):
        # 排除跳过的目录
        dirs[:] = [d for d in dirs if d not in skip_paths and os.path.basename(root) not in skip_paths]
        count += len(files)
    return count

def copy_with_filters(src_dir, dest_dir, pbar=None):
    """复制文件和目录，应用过滤规则"""
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 遍历源目录中的所有项目
    items = os.listdir(src_dir)

    for item in items:
        src_path = os.path.join(src_dir, item)

        # 检查是否应该跳过该路径
        if item in SKIP_PATHS:
            continue

        # 处理路径名称
        processed_item = process_path_name(item)
        dest_path = os.path.join(dest_dir, processed_item)

        # 根据源是文件还是目录进行复制
        if os.path.isdir(src_path):
            # 递归复制子目录
            copy_with_filters(src_path, dest_path, pbar)
        else:
            # 检查是否为JSON或YAML文件
            if src_path.lower().endswith(('.json', '.yaml', '.yml')):
                process_json_yaml_file(src_path, dest_path)
            else:
                # 复制普通文件
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(src_path, dest_path)

            # 更新进度条
            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"处理: {os.path.basename(src_path)}")

if __name__ == "__main__":
    # 检查源目录是否存在
    if not os.path.exists(BENCHMARK_INPUT_DIR):
        print(f"错误: 源目录 '{BENCHMARK_INPUT_DIR}' 不存在")
        exit(1)

    # 计算文件总数，用于进度条
    print("正在计算文件总数...")
    total_files = count_files(BENCHMARK_INPUT_DIR, SKIP_PATHS)
    print(f"共发现 {total_files} 个文件需要处理")

    # 使用tqdm创建进度条
    with tqdm(total=total_files, unit='文件', ncols=100) as pbar:
        print(f"开始将文件从 '{BENCHMARK_INPUT_DIR}' 复制到 '{BENCHMARK_OUTPUT_DIR}'...")
        copy_with_filters(BENCHMARK_INPUT_DIR, BENCHMARK_OUTPUT_DIR, pbar)

    print("复制完成")