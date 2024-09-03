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
    attacker_cls: str = field(default=None)
    attacker_name: str = field(default=None)


class BaseAttacker(abc.ABC):

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

        if messages is None:
            messages = []
        else:
            assert messages[-1]["role"] != "user"

        pass
