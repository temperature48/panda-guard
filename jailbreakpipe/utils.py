# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/1 17:34
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : utils.py
# explain   :

import abc
from typing import Dict, List, Union, Any


def is_user_turn(messages: List[Dict[str, str]]):
    return messages and len(messages) > 0 and messages[-1]["role"] == "user"


def is_assistant_turn(messages: List[Dict[str, str]]):
    return messages and len(messages) > 0 and messages[-1]["role"] == "assistant"
