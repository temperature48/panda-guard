# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/1 17:34
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : utils.py
# explain   :

import abc
from typing import Dict, List, Union, Any, Tuple
import importlib


# Utility function to check if it's the user's turn
def is_user_turn(messages: List[Dict[str, str]]) -> bool:
    return messages and len(messages) > 0 and messages[-1]["role"] == "user"


# Utility function to check if it's the assistant's turn
def is_assistant_turn(messages: List[Dict[str, str]]) -> bool:
    return messages and len(messages) > 0 and messages[-1]["role"] == "assistant"


# Function to load class dynamically based on class name
def load_class(config_cls_name: str) -> Any:
    module = importlib.import_module('jailbreakpipe.role')
    return getattr(module, config_cls_name + 'Config')


def parse_nested_config(config_cls, config_dict: Dict[str, Any]):
    module = importlib.import_module("jailbreakpipe.llms")
    nested_config_dict = {}
    for key, value in config_dict.items():
        if "llm_gen_config" in key:
            nested_config_dict[key] = getattr(module, "LLMGenerateConfig")(**value)
        elif "llm_config" in key:
            nested_config_dict[key] = getattr(module, value["llm_type"] + 'Config')(**value)
        else:
            nested_config_dict[key] = value
    return config_cls(**nested_config_dict)


# Function to convert YAML dictionary into configuration objects
def parse_configs_from_dict(config_dict: Dict[str, Any]):
    # Load the attacker configuration
    attacker_config_dict = config_dict.get("attacker", {})
    AttackerClass = load_class(attacker_config_dict.get("attacker_cls"))
    attacker_config = parse_nested_config(AttackerClass, attacker_config_dict)

    # Load the defender configuration
    defender_config_dict = config_dict.get("defender", {})
    DefenderClass = load_class(defender_config_dict.get("defender_cls"))
    defender_config = parse_nested_config(DefenderClass, defender_config_dict)

    # Load the judge configurations
    judge_configs = []
    judge_config_dicts = config_dict.get("judges", [])
    if judge_config_dicts is not None:
        for judge_config_dict in judge_config_dicts:
            JudgeClass = load_class(judge_config_dict.get("judge_cls"))
            judge_config = parse_nested_config(JudgeClass, judge_config_dict)
            judge_configs.append(judge_config)

    return attacker_config, defender_config, judge_configs
