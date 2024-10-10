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
    """
    Register a defender subclass with the DEFENDERS registry.

    :param defender: Subclass of BaseDefender to register.  要注册的BaseDefender子类
    :return: The registered defender subclass.  返回注册的防御者子类
    """
    DEFENDERS[defender.__name__] = defender
    return defender


def create_defender(config: BaseDefenderConfig) -> BaseDefender:
    """
    Create a defender instance based on the provided configuration.

    :param config: Configuration for the defender.  用于防御者的配置
    :return: An instance of the defender defined by the configuration.  返回配置所定义的防御者实例
    """
    return DEFENDERS[config.defender_cls](config)
