# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/4 15:04
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : attacker_registry.py
# explain   :

from typing import TypeVar
from jailbreakpipe.role.judges import BaseJudge, BaseJudgeConfig


JUDGES: dict[str, type[BaseJudge]] = {}

T = TypeVar("T", bound=BaseJudge)


def register_judge(judge: type[T]) -> type[T]:
    """Register a defense subclass with the defense_hparams module."""
    JUDGES[judge.__name__] = judge
    return judge


def create_judge(config: BaseJudgeConfig) -> BaseJudge:
    return JUDGES[config.judge_cls](config)
