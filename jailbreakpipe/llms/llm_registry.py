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
    """Register a defense subclass with the defense_hparams module."""
    LLMS[llm.__name__] = llm
    return llm


def create_llm(config: Union[BaseLLMConfig, Dict[str, Any]]) -> BaseLLM:
    # if isinstance(config, BaseLLMConfig):
    return LLMS[config.llm_type](config)
    # else:
    #     config = getattr(llms, config["llm_type"] + 'Config')(**config)
    #     return LLMS[config.llm_type](config)
