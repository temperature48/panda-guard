# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/2 19:08
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : register_llm.py
# explain   :

import importlib
from typing import TypeVar, Union, Dict, Any
import jailbreakpipe.llms as llms
from jailbreakpipe.llms.base import BaseLLM, BaseLLMConfig, LLMGenerateConfig

LLMS: dict[str, type[BaseLLM]] = {}

T = TypeVar("T", bound=BaseLLM)


def register_llm(llm: type[T]) -> type[T]:
    """
    Register a subclass of BaseLLM with the LLMS registry.

    :param llm: Subclass of BaseLLM to register.  要注册的BaseLLM的子类
    :return: The registered LLM subclass.  返回注册的LLM子类
    """
    LLMS[llm.__name__] = llm
    return llm


def create_llm(config: Union[BaseLLMConfig, Dict[str, Any]]) -> BaseLLM:
    """
    Create an LLM instance based on the provided configuration.

    :param config: Configuration for LLM, can be a BaseLLMConfig or a dictionary.  用于LLM的配置，可以是BaseLLMConfig或字典形式
    :return: An instance of the LLM defined by the configuration.  返回配置所定义的LLM实例
    """
    return LLMS[config.llm_type](config)
