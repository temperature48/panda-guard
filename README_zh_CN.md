# Jailbreak Pipeline

[English](./README.md) | 简体中文

本仓库包含 `Jailbreak Pipeline` 的源代码，主要用于研究大型语言模型（LLMs）的越狱攻击、防御和评估算法。它基于以下核心原则构建：

- 在 Language Model as a Service (LMaaS) 的背景下，对于服务供应商和用户而言，LLM 应该被视为系统的一部分，而不是独立的实体。因此，研究不仅需要关注 LLM 本身，还需要系统性地研究与之相关的攻击、防御和评估算法。
- “最安全的模型是一个什么都不回答的模型。” 因此，安全性只是模型的一个方面。我们的目标应该是探索安全性与模型能力之间的关系，以及如何在两者之间取得平衡。
- 为了从安全性、能力、效率等多个角度更好地评估不同的模型、攻击和防御算法，我们开发了 `jailbreak-pipeline`，用于评估 LLM 作为系统时的安全性和能力，并探索其在真实场景中的部署。

## 安装

```bash
git clone https://github.com/JailbreakBench/jailbreakbench.git --recurse-submodules
cd jailbreakbench
pip install -r requirements.txt
pip install -e .
```

我们也提供了一个 Docker 镜像，可以通过以下命令获取：

```bash
docker pull 172.18.131.16:23333/floyed/llms@sha256:8b56addb61632316b6ec0862ca3716ca399d139c0485f9b779d14c2aca59b1a9
```

使用此 Docker 镜像并拉取代码库，可以避免大多数依赖问题。

## 使用方法

### Jailbreak Inference

我们提供了一些示例，用于评估 LLM 以及不同的攻击/防御算法。目前，我们设计了两个维度来进行评估，主要是安全性。我们使用 [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) 作为评估数据集。以下脚本用于生成特定模型/攻击/防御配置下的响应：

```bash
cd examples/jailbreak-baselines
python jbb_inference.py \
  --config ../../configs/tasks/jbb.yaml \
  --attack ../../configs/attacks/transfer/gcg.yaml \
  --defense ../../configs/defenses/self_reminder.yaml \
  --llm ../../configs/defenses/llms/llama3.2-1b-it.yaml
```

这些 `*.yaml` 文件用于配置，`--config` 指定默认的任务配置，`--attack`、`--defense` 和 `--llm` 用于重载默认配置。该脚本主要用于生成选定模型、攻击和防御算法下的越狱任务响应。

对于批量实验，我们还提供了脚本 `run_all_inference.py`：

```bash
cd examples/jailbreak-baselines
python run_all_inference.py --max-parallel 8
```

该脚本将遍历 `configs/attacks`、`configs/defenses` 和 `configs/llms` 中的所有配置文件，并进行实验。结果保存在 `benchmarks/jbb` 下，路径形式为 `benchmarks/jbb/{llm}/{attack}/{defense}/`，包含 `results.json`（实验结果）和 `config.yaml`（实验配置）。`--max-parallel 8` 应设置为可用 GPU 的数量，因为负载均衡会在每张 GPU 上运行一个实验。

### Jailbreak Judge

响应生成和评估是独立的过程，以确保资源的最佳利用。因此，通过 `benchmarks/jbb/` 进行的实验将使用 `jbb_eval.py` 进行评估：

```bash
cd examples/jailbreak-baselines
python jbb_eval.py
```

该脚本将评估 `benchmarks/jbb/` 下的所有实验结果。评估结果保存在 `benchmarks/jbb_judged/` 中，结构与 `benchmarks/jbb/` 相似。

### AlpacaEval Inference

为了评估 LLM 的能力，我们使用 [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)。在评估之前，需要按照提供的链接文档安装此库。为了确保评估可用性，因为我们需要使用本地模型作为评估器，建议安装我们内部共享的特定版本。

在这种情况下，我们将防御方法和模型作为一个整体来看待，类似于现实世界中的服务提供。具体的评估脚本如下：

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

运行批量实验：

```bash
cd examples/alpaca_eval-baselines
python run_all_inference.py --max-parallel 8
```

所有结果保存在 `benchmarks/alpaca_eval/{llm}/{defense}/`。

### AlpacaEval Judge

评估过程也与推理分开，可以通过以下命令进行评估：

```bash
cd examples/alpaca_eval-baselines
python alpaca_eval.py
```

结果保存在 `benchmarks/alpaca_eval_judged/` 中，符合 [`AlpacaEval`](https://github.com/tatsu-lab/alpaca_eval) 的官方存储格式。

## 开发

### 要求

首先需要安装 `git lfs`。为了便于代码和数据的分离，减少传输负担，`git lfs` 被用于存储 `benchmarks` 文件夹中的数据：

```bash
git clone https://github.com/JailbreakBench/jailbreakbench.git --recurse-submodules
git lfs install
git lfs pull
```

这样可以将 `benchmarks` 中的数据拉取下来。

### 文档生成

我们使用 `sphinx` 来生成文档。可以使用以下命令生成文档：

```bash
cd docs
sphinx-apidoc -o source/ ../jailbreakpipe/
make html
```

生成的文档将在 `docs/build/html` 下找到。

### 项目结构

源代码位于 `jailbreakpipe` 文件夹下。以下是其结构的简要概述：

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

配置文件位于 `configs` 目录下：

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

用法和评估的示例在 `examples` 中提供：

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

### 代码规范

使用 PyCharm 的默认格式规则。Docstring 使用 'reStructuredText' (rst) 格式，代码风格使用 'black' 格式。以下是一些有用的工具和设置：

- **`black`**: 自动格式化 Python 代码以符合 PEP 8 标准。
- **`flake8`**: 提供潜在代码错误的 linting 并执行风格指南。
- **`isort`**: 自动排序和安排导入，以提高可读性和遵循标准。

## TODO

我们需要包含更多的方法并完成更多评估。候选的方法包括：

### Attackers

- [ ] [GCG](https://arxiv.org/abs/2404.02151)
- [ ] [PAIR](https://arxiv.org/abs/2310.08419)
- [ ] [AutoDan](https://arxiv.org/abs/2310.04451)
- [ ] [Prompt with Random Search](https://arxiv.org/abs/2404.02151)
- [ ] [ArtPrompt](https://aclanthology.org/2024.acl-long.809/)
- [ ] [AdvPrompter](https://arxiv.org/abs/2404.16873)
- [ ] [RainBow Teaming](https://arxiv.org/abs/2402.16822)

### Defenders

- [ ] [GradSafe](https://aclanthology.org/2024.acl-long.30/)
- [ ] [LLM Self Defense](https://arxiv.org/abs/2308.07308)
- [ ] [SafeRLHF](https://arxiv.org/abs/2310.12773) ?
- [ ] [生成后验证]

### Judges

- [ ] 待定

### Dataset

- [ ] 待定

### Task

我们需要选择具体的方法，分配任务给不同的个体，完成这些方法的复现以及评估，然后将结果提交到这个仓库。

## 注意事项

- 不要直接提交代码到 `main` 分支，而是 `checkout` 一个新分支，通过 `pull request` 合并到 `main` 分支。
- 不要直接提交大文件，使用 `git lfs` 进行存储。
- 不要提交敏感信息，比如 API keys 或 tokens。
- 避免提交不必要的文件，如 `__pycache__`、`.idea`、`.vscode`，使用 `.gitignore` 忽略它们。
- 不要提交不必要的代码，如 `print` 或 debug 代码，使用 `logging` 记录日志。
- **紧耦合**：每种方法都要继承对应的抽象类，每个抽象类都要有对应的注册器和配置文件。
- **联系我**：如果有任何问题，尤其是涉及框架设计的，请联系我，一起完善。有些我的考量也不成熟。
