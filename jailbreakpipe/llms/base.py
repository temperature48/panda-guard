# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/8/31 22:00
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : base.py
# explain   :

import abc
from typing import Dict, List, Union, Any, Tuple
from dataclasses import dataclass, field
import concurrent.futures


@dataclass
class LLMGenerateConfig:
    max_n_tokens: int = field(default=None)
    temperature: float = field(default=None)
    logprobs: bool = field(default=False)
    seed: int = field(default=None)


@dataclass
class BaseLLMConfig(abc.ABC):
    llm_type: str = field(default=None)
    model_name: str = field(default=None)


class BaseLLM(abc.ABC):
    def __init__(self, config: BaseLLMConfig):
        self._CLS = config.llm_type
        self._NAME = config.model_name

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.num_requires = 0

    @abc.abstractmethod
    def generate(
            self,
            messages: List[Dict[str, str]],
            config: LLMGenerateConfig
    ) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]]]:
        pass

    @abc.abstractmethod
    def evaluate_log_likelihood(
            self,
            messages: List[Dict[str, str]],
            config: LLMGenerateConfig
    ) -> List[float]:
        pass

    def batch_generate(
            self,
            batch_messages: List[List[Dict[str, str]]],
            config: LLMGenerateConfig,
    ) -> List[Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]]]]:
        configs = [config] * len(batch_messages)
        if config.seed is not None:
            seeds = list(range(config.seed, config.seed + len(batch_messages)))
            for i, c in enumerate(configs):
                c.seed = seeds[i]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.generate, messages, config)
                for config, messages in zip(configs, batch_messages)
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        return results

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.num_requires = 0

    def update(
            self,
            prompt_tokens: int,
            completion_tokens: int,
            num_requires: int,
    ):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.num_requires += num_requires

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    @property
    def avg_tokens(self):
        return self.total_tokens / self.num_requires

