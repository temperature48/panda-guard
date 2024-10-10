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
    """
    Register an attacker subclass with the ATTACKERS registry.

    :param attacker: Subclass of BaseAttacker to register.  要注册的BaseAttacker子类
    :return: The registered attacker subclass.  返回注册的攻击者子类
    """
    ATTACKERS[attacker.__name__] = attacker
    return attacker


def create_attacker(config: BaseAttackerConfig) -> BaseAttacker:
    """
    Create an attacker instance based on the provided configuration.

    :param config: Configuration for the attacker.  用于攻击者的配置
    :return: An instance of the attacker defined by the configuration.  返回配置所定义的攻击者实例
    """
    return ATTACKERS[config.attacker_cls](config)
