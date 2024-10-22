# Jailbreak Pipeline

English | [简体中文](./README_zh_CN.md)

This repository contains the source code for the `Jailbreak Pipeline`, primarily used for researching jailbreak attacks, defenses, and evaluation algorithms for large language models (LLMs). It is built on the following core principles:

- In the context of Language Model as a Service (LMaaS), LLMs should be considered part of a system rather than independent entities for both service providers and users. Therefore, systematic research is needed not only on the LLM itself but also on the associated attack, defense, and evaluation algorithms.
- *The safest model is one that answers nothing.* Therefore, safety is just one aspect of a model. Our goal should be to explore the relationship between safety and model capability and how to balance these two aspects effectively.
- To better evaluate different models, attacks, and defense algorithms from various perspectives, such as safety, capability, and efficiency, we developed the `jailbreak-pipeline`. It is used to assess the safety and capability of LLMs as a system, and explore their deployment in real-world scenarios.

## Installation

```bash
git clone https://github.com/JailbreakBench/jailbreakbench.git --recurse-submodules
cd jailbreakbench
pip install -r requirements.txt
pip install -e .
```

We also provide a Docker image, which can be pulled with the following command:

```bash
docker pull 172.18.131.16:23333/floyed/llms@sha256:8b56addb61632316b6ec0862ca3716ca399d139c0485f9b779d14c2aca59b1a9
```

Using this Docker image and pulling the repository code can help avoid most dependency issues.

## Usage

### Jailbreak Inference

We provide examples to evaluate LLMs along with different attack/defense algorithms. Currently, we designed two dimensions for assessment: primarily safety. We use [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) as the evaluation dataset. The following script is an example for generating responses for specific model/attack/defense configurations:

```bash
cd examples/jailbreak-baselines
python jbb_inference.py \
  --config ../../configs/tasks/jbb.yaml \
  --attack ../../configs/attacks/transfer/gcg.yaml \
  --defense ../../configs/defenses/self_reminder.yaml \
  --llm ../../configs/defenses/llms/llama3.2-1b-it.yaml
```

The `*.yaml` files are used for configuration. The `--config` specifies the default task configuration, while `--attack`, `--defense`, and `--llm` override the default configuration. The script is mainly used to generate jailbreak task responses for the selected model, attack, and defense algorithms.

For batch experiments, we also provide the script `run_all_inference.py`:

```bash
cd examples/jailbreak-baselines
python run_all_inference.py --max-parallel 8
```

This script will iterate through all configuration files in `configs/attacks`, `configs/defenses`, and `configs/llms`, and conduct experiments. The results are saved under `benchmarks/jbb`, with paths in the form `benchmarks/jbb/{llm}/{attack}/{defense}/`, containing `results.json` (experimental results) and `config.yaml` (experiment configuration). `--max-parallel 8` should be set to the number of available GPUs, as load balancing will assign one experiment per GPU.

### Jailbreak Judge

The response generation and evaluation are independent processes to ensure optimal resource utilization. Therefore, experiments conducted under `benchmarks/jbb/` are evaluated using `jbb_eval.py`:

```bash
cd examples/jailbreak-baselines
python jbb_eval.py
```

This script will evaluate all experiment results under `benchmarks/jbb/`. Evaluation results are saved in `benchmarks/jbb_judged/` with a similar structure to `benchmarks/jbb/`.

### AlpacaEval Inference

To evaluate the capabilities of LLMs, we use [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval). Before evaluation, you need to install this library by following the documentation on the provided link. To ensure evaluation availability, since a local model is required as an evaluator, we recommend installing the specific version shared in our internal communication.

In this context, we treat the defense method and the model as a whole, similar to real-world service provision. The evaluation script is as follows:

```bash
cd examples/alpaca_eval-baselines
python alpaca_inference.py \
  --config ../../configs/tasks/alpaca_eval.yaml \
  --llm ../../configs/defenses/llms/phi-3-mini-it.yaml \
  --defense ../../configs/defenses/semantic_smoothllm.yaml \
  --output-dir ../../benchmarks/alpaca_eval \
  --llm-gen ../../configs/defenses/llm_gen/alpaca_eval.yaml \
  --device cuda:7 \
  --max-queries 5 \
  --visible
```

To run batch experiments:

```bash
cd examples/alpaca_eval-baselines
python run_all_inference.py --max-parallel 8
```

All results are saved under `benchmarks/alpaca_eval/{llm}/{defense}/`.

### AlpacaEval Judge

The evaluation process is also separated from the inference, which can be conducted with the following command:

```bash
cd examples/alpaca_eval-baselines
python alpaca_eval.py
```

The results are saved in `benchmarks/alpaca_eval_judged/`, following the official storage format from [`AlpacaEval`](https://github.com/tatsu-lab/alpaca_eval).

## Development

### Requirements

First, you need to install `git lfs`. To facilitate the separation of code and data and reduce transmission load, `git lfs` is used to store data in the `benchmarks` folder:

```bash
git clone https://github.com/JailbreakBench/jailbreakbench.git --recurse-submodules
git lfs install
git lfs pull
```

This will allow you to pull the data in `benchmarks`.

### Documentation Generation

We use `sphinx` to generate documentation. You can generate documentation with the following commands:

```bash
cd docs
sphinx-apidoc -o source/ ../jailbreakpipe/
make html
```

The generated documentation will be found under `docs/build/html`.

### Project Structure

The source code is located under the `jailbreakpipe` folder. Here is a brief overview of the structure:

```
jailbreakpipe
├── __init__.py
├── llms
│   ├── __init__.py
│   ├── base.py
│   ├── hf.py
│   ├── llm_registry.py
│   └── oai.py
├── pipelines
│   ├── __init__.py
│   └── inference.py
├── role
│   ├── __init__.py
│   ├── attacks
│   │   ├── __init__.py
│   │   ├── attacker_registry.py
│   │   ├── base.py
│   │   ├── rewrite.py
│   │   └── transfer.py
│   ├── defenses
│   │   ├── __init__.py
│   │   ├── back_translate.py
│   │   ├── base.py
│   │   ├── defender_registry.py
│   │   ├── icl.py
│   │   ├── paraphrase.py
│   │   ├── perplexity_filter.py
│   │   ├── repe.py
│   │   ├── repe_utils
│   │   ├── rewrite.py
│   │   ├── semantic_smoothing_templates/
│   │   ├── semantic_smoothllm.py
│   │   └── smoothllm.py
│   └── judges
│       ├── __init__.py
│       ├── base.py
│       ├── judge_registry.py
│       ├── llm_based.py
│       └── rule_based.py
└── utils.py
```

Configurations are under the `configs` directory:

```
configs
├── attacks
│   ├── past_tense.yaml
│   └── transfer
│       ├── aim.yaml
│       ├── anti_gpt_v2.yaml
│       ├── better_dan.yaml
│       ├── dev_mode_ranti.yaml
│       ├── dev_mode_v2.yaml
│       ├── future.yaml
│       ├── gcg.yaml
│       ├── original.yaml
│       ├── pair.yaml
│       ├── past.yaml
│       └── prompt_with_random_search.yaml
├── defenses
│   ├── icl.yaml
│   ├── llm_gen
│   │   ├── alpaca_eval.yaml
│   │   └── jbb.yaml
│   ├── llms
│   │   ├── gemma-2-2b-it.yaml
│   │   ├── llama3.1-8b-it.yaml
│   │   ├── llama3.2-1b-it.yaml
│   │   ├── phi-3-mini-it.yaml
│   │   ├── qwen-2-7b-it.yaml
│   │   ├── qwen2.5-1.5b-it.yaml
│   │   ├── qwen2.5-3b-it.yaml
│   │   ├── qwen2.5-7b-it.yaml
│   │   └── reserved/
│   ├── none.yaml
│   ├── paraphrase.yaml
│   ├── perplexity_filter.yaml
│   ├── repe.yaml
│   ├── self_reminder.yaml
│   ├── semantic_smoothllm.yaml
│   └── smoothllm.yaml
├── judges
│   └── judge.yaml
└── tasks
    ├── alpaca_eval.yaml
    └── jbb.yaml
```

Examples for usage and evaluation are provided in `examples`:

```
../examples/
├── alpaca_eval-baselines
│   ├── alpaca_eval.py
│   ├── alpaca_eval_leaderboard.py
│   ├── alpaca_inference.py
│   ├── fails.txt
│   └── run_all_inference.py
└── jailbreak-baselines
    ├── alpaca_eval-baselines
    ├── alpaca_eval_leaderboard.py
    ├── fails.txt
    ├── jbb_eval.py
    ├── jbb_inference.py
    └── run_all_inference.py
```

### Code Style Guidelines

Use the default format rules from PyCharm. Docstrings should follow the 'reStructuredText' (rst) format, and code style should follow the 'black' formatting. Here are some useful tools and settings:

- **`black`**: Automatically formats Python code to conform to the PEP 8 standard.
- **`flake8`**: Provides linting for potential code errors and enforces a style guide.
- **`isort`**: Automatically sorts and arranges imports for readability and adherence to standards.

## TODO

We need to include more methods and complete more evaluations. The candidates include:

### Attackers

- [ ] [GCG](https://arxiv.org/abs/2404.02151)
- [ ] [PAIR](https://arxiv.org/abs/2310.08419)
- [ ] [AutoDan](https://arxiv.org/abs/2310.04451)
- [ ] [Prompt with Random Search](https://arxiv.org/abs/2404.02151)
- [ ] [ArtPrompt](https://aclanthology.org/2024.acl-long.809/)
- [ ] [AdvPrompter](https://arxiv.org/abs/2404.16873)
- [ ] [RainBow Teaming](https://arxiv.org/abs/2402.16822)
- [ ] [ICA](http://arxiv.org/abs/2310.06387)
- [ ] [GPT-4-Cipher](http://arxiv.org/abs/2308.06463)
- [ ] [Backtranslation](http://arxiv.org/abs/2308.06259)
- [ ] [Multilingual](http://arxiv.org/abs/2310.06474)
- [ ] [GPTFUZZER](https://arxiv.org/abs/2309.10253v4)

### Defenders

- [ ] [GradSafe](https://aclanthology.org/2024.acl-long.30/)
- [ ] [LLM Self Defense](https://arxiv.org/abs/2308.07308)
- [ ] [SafeRLHF](https://arxiv.org/abs/2310.12773) ?
- [ ] [ICD](http://arxiv.org/abs/2310.06387)
- [ ] [Post-Generation Validation]

### Judges

- [ ] TBD

### Datasets

- [ ] TBD

### Tasks

We need to select specific methods, assign tasks to different individuals, complete reproductions of these methods, and conduct evaluations, then submit the results to this repository.

## Notes

- Do not submit code directly to the `main` branch. Instead, `checkout` a new branch and merge to `main` via `pull request`.
- Do not submit large files directly; use `git lfs` for storage.
- Do not submit sensitive information, such as API keys or tokens.
- Avoid submitting unnecessary files like `__pycache__`, `.idea`, `.vscode`; use `.gitignore` to ignore them.
- Do not submit unnecessary code like `print` or debug statements; use `logging` for recording logs.
- **Tight Coupling**: Each method must inherit from its respective abstract class, and every abstract class should have a corresponding registry and configuration file.
- **Contact Me**: If you have any questions, especially those involving framework design, please reach out so we can improve together. Some of my considerations may also need refinement.

