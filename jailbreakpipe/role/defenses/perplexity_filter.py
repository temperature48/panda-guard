# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/3 22:39
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : perplexity_filter.py
# explain   :
from dataclasses import dataclass, field
from typing import List, Dict
import math
import warnings

import numpy as np

from jailbreakpipe.role.defenses import BaseDefender, BaseDefenderConfig, REJECT_RESPONSE
from jailbreakpipe.role.defenses.defender_registry import register_defender
from jailbreakpipe.llms import BaseLLMConfig, LLMGenerateConfig, create_llm
from jailbreakpipe.utils import is_user_turn


@dataclass
class PerplexityFilterDefenderConfig(BaseDefenderConfig):
    defender_cls: str = field(default="PerplexityFilterDefender")
    threshold: float = field(default=500)
    filter_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    filter_llm_gen_config: LLMGenerateConfig = field(default_factory=LLMGenerateConfig)


@register_defender
class PerplexityFilterDefender(BaseDefender):
    """
    Baseline Defenses for Adversarial Attacks Against Aligned Language Models
    Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang,
    Micah Goldblum, Aniruddha Saha, Jonas Geiping,
    Tom Goldstein https://arxiv.org/abs/2309.00614
    """

    def __init__(self, config: PerplexityFilterDefenderConfig):
        super().__init__(config)
        # print(f'Using threshold {config.threshold} for backtranslation')
        self.filter_llm = create_llm(config.filter_llm_config)
        self.filter_llm_gen_config = config.filter_llm_gen_config
        self.threshold = config.threshold

    def defense(
            self,
            messages: List[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        assert is_user_turn(messages), "It must be the user's turn to perform defense."

        # Evaluate log likelihoods for the messages
        logprobs = self.filter_llm.evaluate_log_likelihood(messages, self.filter_llm_gen_config)

        # Calculate perplexity
        perplexity = self._calculate_perplexity(logprobs)
        # print(perplexity)
        # Check if perplexity exceeds the threshold
        if perplexity > self.threshold:
            messages.append({"role": "assistant", "content": REJECT_RESPONSE})
            return messages

        return super().defense(messages)

    def _calculate_perplexity(self, logprobs: List[float]) -> float:
        # Compute the average log probability
        avg_log_prob = np.mean(logprobs)

        # Calculate perplexity
        perplexity = math.exp(-avg_log_prob)
        return perplexity
