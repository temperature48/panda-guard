# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/3 12:19
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : paraphrase.py
# explain   :
from dataclasses import dataclass, field
from typing import List, Dict
import warnings

from jailbreakpipe.role.defenses import BaseDefender, BaseDefenderConfig
from jailbreakpipe.role.defenses.defender_registry import register_defender
from jailbreakpipe.llms import BaseLLMConfig, LLMGenerateConfig, create_llm


@dataclass
class ParaphraseDefenderConfig(BaseDefenderConfig):
    defender_cls: str = field(default="ParaphraseDefender")
    paraphrase_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    paraphrase_llm_gen_config: LLMGenerateConfig = field(default_factory=lambda: LLMGenerateConfig(max_n_tokens=512))
    paraphrase_prompt: str = field(default='paraphrase the following paragraph: \n"{prompt}"\n\n')


@register_defender
class ParaphraseDefender(BaseDefender):
    """
    Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. 2023.
    Baseline defenses for adversarial attacks against aligned language models.
    arXiv preprint arXiv:2309.00614.
    """
    def __init__(self, config: ParaphraseDefenderConfig):
        super().__init__(config)
        self.paraphrase_llm = create_llm(config.paraphrase_llm_config)
        self.paraphrase_llm_gen_config = config.paraphrase_llm_gen_config
        self.paraphrase_prompt = config.paraphrase_prompt

    def defense(
            self,
            messages: List[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:

        assert messages, "Messages cannot be empty."

        prompt = messages[-1]['content']
        paraphrase_prompt = self._paraphrase(prompt)

        if "\n" in paraphrase_prompt:
            warnings.warn("""A \\n character is found in the output of paraphrase model and the content after \\n is removed.""")
            paraphrase_prompt = "\n".join(paraphrase_prompt.split('\n')[1:])

        messages[-1]['content'] = paraphrase_prompt

        return super().defense(messages)

    def _paraphrase(self, prompt: str) -> str:
        paraphrase_prompt = self.paraphrase_prompt.format(prompt=prompt)

        output = self.paraphrase_llm.generate(
            [{"role": "user", "content": paraphrase_prompt}],
            self.paraphrase_llm_gen_config
        )[-1]['content']

        paraphrase_prompt = output.strip().strip(']').strip('[')

        return paraphrase_prompt
