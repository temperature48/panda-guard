# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/1 12:57
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : base.py
# explain   :

import abc
from typing import Dict, List, Union, Any
from dataclasses import dataclass, field


@dataclass
class BaseAttackerConfig(abc.ABC):
    """
    Configuration for the Base Attacker.

    :param attacker_cls: Class of the attacker.  攻击者的类型
    :param attacker_name: Name of the attacker.  攻击者的名称
    """
    attacker_cls: str = field(default=None)
    attacker_name: str = field(default=None)


class BaseAttacker(abc.ABC):
    """
    Abstract Base Class for Attacker.

    :param config: Configuration for the attacker.  攻击者的配置
    """

    def __init__(
            self,
            config: BaseAttackerConfig
    ):
        self._CLS = config.attacker_cls
        self._NAME = config.attacker_name

    @abc.abstractmethod
    def attack(
            self,
            messages: List[Dict[str, str]],
            **kwargs
    ) -> List[Dict[str, str]]:
        """
        Abstract method to execute an attack on a sequence of messages.

        :param messages: List of input messages.  输入的消息列表
        :param kwargs: Additional parameters for the attack.  攻击的额外参数
        :return: Modified list of messages after the attack.  攻击后返回的更改消息列表
        """
        if messages is None:
            messages = []
        else:
            assert messages[-1]["role"] != "user"

        pass
