# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/4 15:04
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : attacker_registry.py
# explain   :

from typing import TypeVar
from jailbreakpipe.role.attacks import BaseAttacker, BaseAttackerConfig


ATTACKERS: dict[str, type[BaseAttacker]] = {}

T = TypeVar("T", bound=BaseAttacker)


def register_attacker(attacker: type[T]) -> type[T]:
    """Register a defense subclass with the defense_hparams module."""
    ATTACKERS[attacker.__name__] = attacker
    return attacker


def create_attacker(config: BaseAttackerConfig) -> BaseAttacker:
    return ATTACKERS[config.attacker_cls](config)
