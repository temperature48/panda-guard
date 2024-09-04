# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/4 15:04
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : attacker_registry.py
# explain   :

from typing import TypeVar
from jailbreakpipe.role.defenses import BaseDefender, BaseDefenderConfig


DEFENDERS: dict[str, type[BaseDefender]] = {}

T = TypeVar("T", bound=BaseDefender)


def register_defender(defender: type[T]) -> type[T]:
    """Register a defense subclass with the defense_hparams module."""
    DEFENDERS[defender.__name__] = defender
    return defender


def create_defender(config: BaseDefenderConfig) -> BaseDefender:
    return DEFENDERS[config.defender_cls](config)
