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

        # # Generate the response using the target LLM
        # paraphrase_response = self.target_llm.generate(
        #     [{"role": "user", "content": paraphrase_prompt}],
        #     self.target_llm_gen_config
        # )[-1]['content']
        #
        # # Append the generated response to the messages list
        # messages.append({
        #     "role": "assistant",
        #     "content": paraphrase_response
        # })
        messages[-1]['content'] = paraphrase_prompt

        return super().defense(messages)

    def _paraphrase(self, prompt: str, verbose: bool = False) -> str:
        paraphrase_prompt = f'paraphrase the following paragraph: \n"{prompt}"\n\n'

        if verbose:
            print('Asking the model to paraphrase the prompt:')
            print(paraphrase_prompt)

        output = self.paraphrase_llm.generate(
            [{"role": "user", "content": paraphrase_prompt}],
            self.paraphrase_llm_gen_config
        )[-1]['content']

        paraphrase_prompt = output.strip().strip(']').strip('[')

        if verbose:
            print('Paraphrased prompt:', paraphrase_prompt)

        return paraphrase_prompt
