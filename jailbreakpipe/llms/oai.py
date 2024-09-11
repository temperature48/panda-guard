# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/8/31 22:01
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : oai.py
# explain   :

import abc
import os
from typing import Dict, List, Union, Any, Tuple
from dataclasses import dataclass, field
import concurrent.futures
import torch
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer

from jailbreakpipe.llms.llm_registry import register_llm
from jailbreakpipe.llms import BaseLLM, BaseLLMConfig, LLMGenerateConfig


@dataclass
class OpenAiLLMConfig(BaseLLMConfig):
    llm_type: str = field(default="OpenAiLLM")
    model_name: str = field(default=None)
    base_url: str = field(default=None)
    api_key: str = field(default="KEY HERE")


@dataclass
class OpenAiChatLLMConfig(BaseLLMConfig):
    llm_type: str = field(default="OpenAiChatLLM")
    model_name: str = field(default=None)
    base_url: str = field(default=None)
    api_key: str = field(default="KEY HERE")


@register_llm
class OpenAiChatLLM(BaseLLM):
    def __init__(
            self,
            config: OpenAiLLMConfig
    ):
        super().__init__(config)
        self.client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    def generate(
            self,
            messages: List[Dict[str, str]],
            config: LLMGenerateConfig
    ) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]]]:

        response = self.client.chat.completions.create(
            model=self._NAME,
            messages=messages,
            max_tokens=config.max_n_tokens,
            temperature=config.temperature,
            logprobs=config.logprobs,
            seed=config.seed,
        )
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})

        self.update(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            1,
        )

        if config.logprobs:
            logs = []
            for c in response.choices[0].logprobs.content:
                logs.append(c.logprob)
            return messages, logs

        return messages

    def evaluate_log_likelihood(
            self,
            messages: List[Dict[str, str]],
            config: LLMGenerateConfig
    ) -> List[float]:
        raise NotImplementedError("OpenAI Chat does not support log likelihood evaluation.")


@register_llm
class OpenAiLLM(BaseLLM):
    def __init__(
            self,
            config: OpenAiLLMConfig
    ):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            # self.model_name.replace('3.1', '3'),
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=True
        )
        self.client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    def generate(
            self,
            messages: List[Dict[str, str]],
            config: LLMGenerateConfig
    ) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]]]:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = self.client.completions.create(
            model=self._NAME,
            prompt=prompt,
            max_tokens=config.max_n_tokens,
            temperature=config.temperature,
            logprobs=config.logprobs,
        )
        content = response.choices[0].text
        messages.append({"role": "assistant", "content": content})
        # print(messages)

        self.update(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            1,
        )
        # print(response)
        if config.logprobs:
            logs = response.choices[0].logprobs.token_logprobs
            return messages, logs

        return messages

    def evaluate_log_likelihood(
            self,
            messages: List[Dict[str, str]],
            config: LLMGenerateConfig
    ) -> List[float]:
        # print(messages)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = self.client.completions.create(
            model=self._NAME,
            prompt=prompt,
            max_tokens=0,
            temperature=config.temperature,
            logprobs=1,
            echo=True
        )
        logprobs = response.choices[0].logprobs.token_logprobs

        # print(self.tokenizer(messages[-1]['content']))
        self.update(response.usage.prompt_tokens, 0, 1)

        return logprobs[-len(self.tokenizer(messages[-1]['content']).input_ids):]


if __name__ == '__main__':
    from jailbreakpipe.llms import LLMS
    print(LLMS)

    llm_gen_config = LLMGenerateConfig(
        max_n_tokens=128,
        temperature=1.,
        logprobs=True,
        seed=42
    )

    config = OpenAiLLMConfig(
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        base_url='http://172.18.129.80:8000/v1'
    )
    llm = OpenAiLLM(config)

    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    results = llm.evaluate_log_likelihood(messages, llm_gen_config)
    print(results, len(results))

    # batch_messages = [
    #     [{"role": "user", "content": "Hello, how are you?"}],
    #     [{"role": "user", "content": "What is the weather like today?"}],
    #     [{"role": "user", "content": "Tell me a joke."}],
    # ]
    #
    # results = llm.batch_generate(
    #     batch_messages,
    #     llm_gen_config
    # )
    # for result in results:
    #     print(result)
    #
    # print(llm.avg_tokens, llm.total_tokens)
