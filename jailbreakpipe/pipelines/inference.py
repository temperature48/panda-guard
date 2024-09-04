# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/4 14:37
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : inference.py
# explain   :

import abc
from typing import Dict, List, Union, Any
from dataclasses import dataclass, field

from jailbreakpipe.role.judges.rule_based import BaseJudge
from jailbreakpipe.llms import BaseLLM, BaseLLMConfig
from jailbreakpipe.role import create_attacker, create_defender, create_judge, BaseAttacker, BaseDefender, \
    TransferAttacker
from jailbreakpipe.role import BaseAttackerConfig, BaseDefenderConfig, BaseJudgeConfig


@dataclass
class InferPipelineConfig:
    attacker_config: BaseAttackerConfig = field(default=None)
    defender_config: BaseDefenderConfig = field(default=None)
    judge_configs: List[BaseJudgeConfig] = field(default_factory=list)


def llm_register(attr: Union[BaseAttacker, BaseDefender, BaseJudge]) -> List[BaseLLM]:
    results = []
    for name, value in vars(attr).items():
        if isinstance(value, BaseLLM):
            results.append(value)
    return results


class InferPipeline:
    def __init__(
            self,
            config: InferPipelineConfig,
            verbose: bool = False
    ):
        self.attacker = create_attacker(config.attacker_config)
        self.defender = create_defender(config.defender_config)
        self.judges = [create_judge(judge_config) for judge_config in config.judge_configs]

        self.attack_llms = llm_register(self.attacker)
        self.defense_llms = llm_register(self.defender)
        self.judges_llms = [llm_register(judge) for judge in self.judges]

        self.verbose = verbose

    def __call__(self, messages: List[Dict[str, str]], request: str, **kwargs) -> Dict[str, Any]:

        if isinstance(self.attacker, TransferAttacker):
            assert "request_reformulated" in kwargs

        self.log(messages, "[INPUT]")

        attack = self.attacker.attack(messages, request_reformulated=kwargs.get("request_reformulated", None))
        self.log(attack, "[ATTACK]")

        defense = self.defender.defense(attack)
        self.log(defense, "[DEFENSE]")

        judge_results = {judge._NAME: judge.judge(defense, request) for judge in self.judges}

        usage = self.calc_tokens()

        return {
            "messages": defense,
            "judges": judge_results,
            "usage": usage
        }

    def calc_tokens(self) -> Dict[str, Dict[str, int]]:
        attacker_prompt_tokens = sum(llm.prompt_tokens for llm in self.attack_llms)
        attacker_completion_tokens = sum(llm.completion_tokens for llm in self.attack_llms)

        defender_prompt_tokens = sum(llm.prompt_tokens for llm in self.defense_llms)
        defender_completion_tokens = sum(llm.completion_tokens for llm in self.defense_llms)

        judge_prompt_tokens = sum(llm.prompt_tokens for llms in self.judges_llms for llm in llms)
        judge_completion_tokens = sum(llm.completion_tokens for llms in self.judges_llms for llm in llms)

        return {
            "attacker": {"prompt_tokens": attacker_prompt_tokens, "completion_tokens": attacker_completion_tokens},
            "defender": {"prompt_tokens": defender_prompt_tokens, "completion_tokens": defender_completion_tokens},
            "judge": {"prompt_tokens": judge_prompt_tokens, "completion_tokens": judge_completion_tokens}
        }

    def reset(self):
        for llm in self.attack_llms + self.defense_llms + [llm for llms in self.judges_llms for llm in llms]:
            llm.reset()

    def log(self, messages: List[Dict[str, str]], stage: str = None):
        if self.verbose:
            print(stage)
            print(messages)
            print()


