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
    """
    Register a judge subclass with the registry. 将 Judge 子类注册到注册表中

    :param judge: The judge class to be registered. 需要注册的 Judge 类
    :return: The registered judge class. 返回已注册的 Judge 类
    """
    JUDGES[judge.__name__] = judge
    return judge


def create_judge(config: BaseJudgeConfig) -> BaseJudge:
    """
    Create an instance of the registered judge class based on the given configuration. 根据给定的配置创建已注册 Judge 类的实例

    :param config: Configuration for the judge. Judge 的配置
    :return: An instance of the judge. Judge 的实例
    """
    return JUDGES[config.judge_cls](config)
