# InstructDiff: Domain-Adaptive Data Selection for Efficient LLM Fine-Tuning

## 中文简介

本仓库实现了 InstructDiff 的可复现实验流程，包括：
- **Warmup Calibration**：使用小规模随机子集校准基础模型
- **Distribution-Aware Selection**：计算 base 与 calibrated 的 NLL/熵差分，进行双向 NLL 过滤与差分熵排序
- **Iterative Refinement**：以上流程可迭代重复，形成逐轮筛选与训练

论文图示：

![Pipeline](acl26_instructDiff/figs/pipeline.png)

本仓库保留 `eval/` 与 `data/` 作为对照与复现入口，所有核心代码统一放入 `src/`，配置放在 `configs/`。

## English Overview

This repository provides a reproducible implementation of InstructDiff with:
- **Warmup Calibration** on a random subset
- **Distribution-Aware Selection** using NLL/entropy differences and bi-directional NLL filtering
- **Iterative Refinement** to repeat selection and training across rounds

Paper figure:

![Pipeline](acl26_instructDiff/figs/pipeline.png)

Core code lives in `src/`, configs in `configs/`, and evaluation/data are preserved in `eval/` and `data/`.

## Installation

```bash
pip install -e .
```

Dependencies are listed in `requirements.txt`.

## Quick Start (Med 1k → 100)

```bash
# sample 100 from first 1k
instdiff sample --config configs/med/sample_med_1k_100.yaml

# train on 100 samples
instdiff train --config configs/med/train_med_100.yaml

# run iterative pipeline (2 rounds)
instdiff pipeline --config configs/med/pipeline_med_iter.yaml
instdiff pipeline --config configs/med/pipeline_med_iter2.yaml
```

Outputs are written under `runs/` (auto-created).

## Experiment Results

### Med domain (Llama-2-7b-hf)

| Exp ID | Data | Setting | Metric | Value |
|--------|------|---------|--------|-------|
| med_score_001 | med 10k (sample=100) | base scoring | avg_nll | 2.0265 |
| med_score_001 | med 10k (sample=100) | base scoring | avg_entropy | 1.9091 |
| med_iter_001 | med 1k → 100 | iter-1 | pipeline | done |
| med_iter_002 | med 1k → 100 | iter-2 | pipeline | done |

## Repository Structure

```
Data-Selection/
├── acl26_instructDiff/         # paper source (kept)
├── configs/                    # YAML configs
├── data/                       # datasets
├── eval/                       # evaluation (kept)
├── src/                        # core code
│   ├── instdiff/               # CLI + pipeline
│   └── instdiff_tools/         # analysis/plot tools
├── requirements.txt
├── setup.py
└── README.md
```

## Notes

- `acl26_instructDiff` is preserved as the paper source.
- `eval/` and `data/` remain unchanged for reproduction.
- Tools are under `src/instdiff_tools/analysis/`.

# Instruct-diff
