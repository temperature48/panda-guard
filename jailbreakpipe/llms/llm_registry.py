# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/2 19:08
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : register_llm.py
# explain   :

from typing import TypeVar
from jailbreakpipe.llms.base import BaseLLM, BaseLLMConfig, LLMGenerateConfig


LLMS: dict[str, type[BaseLLM]] = {}

T = TypeVar("T", bound=BaseLLM)


def register_llm(llm: type[T]) -> type[T]:
    """Register a defense subclass with the defense_hparams module."""
    LLMS[llm.__name__] = llm
    return llm


def create_llm(config: BaseLLMConfig) -> BaseLLM:
    return LLMS[config.llm_type](config)
