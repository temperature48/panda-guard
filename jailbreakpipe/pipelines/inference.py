# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/4 14:37
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : inference.py
# explain   : Inference pipeline for executing attack and defense.

import abc
from typing import Dict, List, Union, Any
from dataclasses import dataclass, field

from jailbreakpipe.llms import BaseLLM, BaseLLMConfig
from jailbreakpipe.role import create_attacker, create_defender, create_judge, BaseAttacker, BaseDefender, \
    TransferAttacker
from jailbreakpipe.role import BaseAttackerConfig, BaseDefenderConfig


@dataclass
class InferPipelineConfig:
    """
    Configuration class for the inference pipeline.

    配置类，用于推理流程。

    :param attacker_config: Configuration for the attacker.
                            攻击者的配置。
    :param defender_config: Configuration for the defender.
                            防御者的配置。
    """
    attacker_config: BaseAttackerConfig = field(default=None)
    defender_config: BaseDefenderConfig = field(default=None)


def llm_register(attr: Union[BaseAttacker, BaseDefender]) -> List[BaseLLM]:
    """
    Register LLMs from an attacker or defender.

    从攻击者或防御者中注册LLM。

    :param attr: The object containing LLM attributes.
                 包含LLM属性的对象。
    :return: List of registered LLMs.
             注册的LLM列表。
    """
    results = []
    for name, value in vars(attr).items():
        if isinstance(value, BaseLLM):
            results.append(value)
    return results


class InferPipeline:
    """
    Inference pipeline for handling attacks and defenses.

    处理攻击和防御的推理流程。

    :param config: Configuration for the inference pipeline.
                   推理流程的配置。
    :param verbose: Whether to enable verbose logging.
                    是否启用详细日志。
    """

    def __init__(
            self,
            config: InferPipelineConfig,
            verbose: bool = False
    ):
        self.attacker = create_attacker(config.attacker_config)
        self.defender = create_defender(config.defender_config)

        # Register LLMs for attacker and defender.
        # 为攻击者和防御者注册LLM。
        self.attack_llms = llm_register(self.attacker)
        self.defense_llms = llm_register(self.defender)

        self.verbose = verbose

    def __call__(self, messages: List[Dict[str, str]], request: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the inference pipeline.

        执行推理流程。

        :param messages: Input messages for the pipeline.
                         流程的输入消息。
        :param request: Request content for the attacker.
                        攻击者的请求内容。
        :param kwargs: Additional keyword arguments, e.g., reformulated request.
                       额外的关键字参数，例如重构后的请求。
        :return: A dictionary containing processed messages and token usage.
                 包含处理后的消息和Token使用量的字典。
        """
        # Check if the attacker is a TransferAttacker.
        # 检查攻击者是否为TransferAttacker。
        if isinstance(self.attacker, TransferAttacker):
            assert "request_reformulated" in kwargs

        self.log(messages, "[INPUT]")

        # Perform attack phase.
        # 执行攻击阶段。
        attack = self.attacker.attack(messages, request_reformulated=kwargs.get("request_reformulated", None))
        self.log(attack, "[ATTACK]")

        # Perform defense phase.
        # 执行防御阶段。
        defense = self.defender.defense(attack)
        self.log(defense, "[DEFENSE]")

        # Calculate token usage.
        # 计算Token使用量。
        usage = self.calc_tokens()

        return {
            "messages": defense,
            "usage": usage
        }

    def calc_tokens(self) -> Dict[str, Dict[str, int]]:
        """
        Calculate the token usage for attacker and defender.

        计算攻击者和防御者的Token使用量。

        :return: Dictionary with token usage details.
                 包含Token使用详情的字典。
        """
        attacker_prompt_tokens = sum(llm.prompt_tokens for llm in self.attack_llms)
        attacker_completion_tokens = sum(llm.completion_tokens for llm in self.attack_llms)

        defender_prompt_tokens = sum(llm.prompt_tokens for llm in self.defense_llms)
        defender_completion_tokens = sum(llm.completion_tokens for llm in self.defense_llms)

        return {
            "attacker": {"prompt_tokens": attacker_prompt_tokens, "completion_tokens": attacker_completion_tokens},
            "defender": {"prompt_tokens": defender_prompt_tokens, "completion_tokens": defender_completion_tokens},
        }

    def reset(self):
        """
        Reset the state of all LLMs in the pipeline.

        重置流程中所有LLM的状态。
        """
        for llm in self.attack_llms + self.defense_llms:
            llm.reset()

    def log(self, messages: List[Dict[str, str]], stage: str = None):
        """
        Log messages for each stage if verbose is enabled.

        如果启用了详细日志记录，则记录每个阶段的消息。

        :param messages: Messages to be logged.
                         需要记录的消息。
        :param stage: The stage of the pipeline for logging.
                      用于日志记录的流程阶段。
        """
        if self.verbose:
            print(stage)
            print(messages)
            print()
