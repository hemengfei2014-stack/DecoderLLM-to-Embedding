# Decoder-only LLM 转换为 Embedding

## 目录
- [项目目标](#项目目标)
- [动机与背景](#动机与背景)
- [思路总览](#思路总览)
- [支持现状](#支持现状)
- [训练流程总览](#训练流程总览)
- [数据下载与预处理](#数据下载与预处理复用-f2llm)
- [执行训练任务](#执行训练任务)
- [效果评测](#效果评测mteb--t2retrieval)

## 项目目标
- 将通用解码式大模型（LLM）转化为向量表征（Embedding），用于检索与匹配等任务。
- 提供工程化、可扩展、可复现的工具箱，支持系统性验证与对比不同转换方案的有效性与稳定性。

## 动机与背景
- 资源—能力正反馈：LLM 在多项任务上取得显著成功，持续吸引大量社会与产业资源投入，形成资源驱动能力加速迭代的正向循环。基于 LLM 构建或与 LLM 深度集成的业务及技术方案，能够随着模型能力的持续提升而同步获益，实现水涨船高的效果。
- 工程化缺口与项目定位：尽管“将 LLM 转换为 Embedding”已有一些研究探索，现有实现多为论文附带的原型代码，侧重验证思路而非工程复用，也有些模型也并不开放源码，缺少统一、可扩展、可复现的工具箱。本项目旨在填补该空缺，提供面向工程的工具箱以支持不同 LLM 的转换、评估与对比。

## 思路总览
- 当前业界将 Decoder-only LLM 转化为 Embedding 的方案主要围绕以下 5 个环节展开：

| 核心环节 | 关键策略 | 详细说明与代表案例 |
| :--- | :--- | :--- |
| **1. 架构改造**<br>(Architecture Adaptation) | **零结构改动** (推荐) | 保持 Causal Mask 不变，仅靠指令与池化适配。<br>_案例：F2LLM, Qwen3-Embedding_ |
| | **轻度改动** | 在模型顶部增加 Linear 或 MLP 作为表征头。 |
| | **深度改动** (高性能) | 移除 Causal Mask，启用双向注意力 (Bidirectional Attention)。<br>_案例：NV-Embed, Jina v4_ |
| **2. 表征提取**<br>(Pooling Strategy) | **EOS / Last Token** | 取最后一个有效 Token，常配合“零结构改动”使用。 |
| | **Mean / Weighted** | 取所有 Token 均值或加权，适合双向注意力模型。 |
| | **Attention Pooling** | 引入可学习 Query 向量捕捉重点信息。 |
| | **Skip Instruction** | 池化时跳过 `Instruct:` 前缀，仅聚合正文 Token 以减少干扰。 |
| | **其他策略** | 句首插入 `[CLS]`、多向量 (ColBERT)、稀疏向量 (Sparse) 等。 |
| **3. 训练目标**<br>(Loss & Objectives) | **对比学习** (核心) | InfoNCE Loss，拉近正样本，推远负样本 (含 In-batch/Hard Negatives)。 |
| | **Matryoshka (MRL)** | 支持弹性维度裁剪 (如 768/512/256 维均可用)。 |
| | **均匀性正则** | Spread-out/Uniformity 强制向量均匀分布，防止表达塌缩 (Anisotropy)。 |
| **4. 数据工程**<br>(Data Engineering) | **指令微调** | 使用 `Instruct: ...` 模板激发 LLM 理解当前任务（检索、分类、聚类等） |
| | **难负样本** | 挖掘“似是而非”的样本提升分辨力。 |
| | **合成与多任务** | 利用 LLM 生成数据，混合检索、STS、聚类等多任务数据。 |
| **5. 训练流程**<br>(Training Pipeline) | **分阶段训练** | **Phase A**: 大规模弱监督预训练 (对齐空间)<br>**Phase B**: 高质量监督微调 (提升精度)<br>**Phase C**: 权重合并/蒸馏 (可选优化) |
| | **LoRA 适配** | 降低显存开销，兼容上述所有策略的高效微调方案。 |


## 支持现状

- 本项目目前实现与评估主要聚焦第 1、2、3 点（架构改造、表征提取、损失设计）的工程化落地与对比。
- 4、5 点（数据工程与分阶段训练流程）暂时复用 [F2LLM](https://github.com/codefuse-ai/CodeFuse-Embeddings/tree/main/F2LLM) 项目，等1、2、3 开发完后，会重点优化4、5。

| 环节 | 已经支持 | 将要支持 |
| --- | --- | --- |
| 架构改造 | Bidirectional Attention（已支持 deepseek v3、qwen3 架构） | — |
| 表征提取策略 | EOS Pooling、Mean Pooling、Weighted Mean Pooling、Skip Instruction | Multi-vector、Sparse Embedding、Attention Pooling |
| 训练目标与损失设计 | 对比学习（In-batch Negatives + Hard Negatives）、MRL、Spread-out | — |
| 分阶段训练流程 | LoRA | — |

## 训练流程总览

1. [数据下载与预处理](#数据下载与预处理复用-f2llm)：复用 F2LLM 数据与分词脚本，生成统一 parquet。
2. [执行训练任务](#执行训练任务)：依据配置文件运行训练，可选启用 MRL、Spread-out、LoRA、双向注意力等组件。
3. [效果评测](#效果评测mteb--t2retrieval)：使用 MTEB/T2Retrieval 进行指标评测与对比。

## 数据下载与预处理（复用 F2LLM）

- 数据来源：复用 F2LLM 项目训练数据集与预处理脚本。
- 下载地址：`https://huggingface.co/datasets/codefuse-ai/F2LLM`
- 预处理脚本：`tokenize_data_qwen.py` 脚本使用 Qwen3 的 tokenizer，如果用别的 LLM 转 embedding，则需要修改为对应 LLM 的 tokenizer。
- 目的：将原始文本字段（`query`、`passage`、`negative_i`）分词为 `input_ids`，并统一输出为 parquet，供训练与评测使用。

### 步骤 1：下载数据

- 使用 `git lfs` 或 `huggingface-hub` 下载数据到本地（示例采用 `huggingface-cli`）：

```
huggingface-cli download codefuse-ai/F2LLM --repo-type dataset --local-dir ./training_data
```

- 说明：上述命令会将数据集的 parquet 文件放置到 `./training_data/` 下（按数据集划分目录）。


### 步骤 2：执行预处理（分词）

- 在项目目录下运行分词脚本：

```
python tokenize_data_qwen.py
```

- 输出：分词后的 parquet 文件写入 `data_tokenized_qwen/{ds_name}`。


### 训练数据简介

- 数据规模与构成：
  - 总样本数约 `5,933,988`，训练集约 `5,874,624`，验证集约 `59,364`（详见 `data_stats.json`）。
  - 类型分布：检索约 `4,918,949`，聚类约 `822,001`，分类约 `193,038` 条目（训练/测试按 0.99/0.01 划分）。
- 任务覆盖：
  - 检索：`MS MARCO`、`NaturalQuestions`、`HotpotQA`、`CNN/DailyMail`、`TriviaQA`、`PAQ` 等，含大量难负样本（`neg_text_columns=24`）。
  - 聚类：`arXiv`、`bioRxiv`、`medRxiv`、`banking77`、`massive`、`twentynewsgroups` 等语料的句/段聚类切分。
  - 分类：`IMDB`、`Amazon Polarity`、`CoLA`、`Toxic Conversations` 等，负例列为 1（`neg_text_columns=1`）。
  

## 执行训练任务

### 配置字段说明（以 `configs/config.full.json` 为例）

#### 路径与实验标识

| 字段 | 作用 |
| --- | --- |
| `model_path` | 底座模型标识/路径 |
| `experiment_id` | 实验名称，拼接输出与日志路径 |
| `train_data_path` | 分词后 parquet 数据目录 |
| `output_dir` / `tb_dir` / `cache_dir` | 输出/TensorBoard/缓存目录 |

#### 数据加载与采样

| 字段 | 作用 |
| --- | --- |
| `num_hard_neg` | 每样本难负例数；分类固定为 1 |

#### 序列长度与拼接

| 字段 | 作用 |
| --- | --- |
| `max_seq_length` | 拼接与右侧 padding 的最大长度 |

#### 训练步数与频率

| 字段 | 作用 |
| --- | --- |
| `train_batch_size` | batch 大小，亦用于 deepspeed micro-batch |
| `train_epochs` | 训练轮次；`train_steps<0` 时按轮次计算总步数 |
| `checkpointing_steps` | 步进保存 checkpoint 的频率 |
| `validation_steps` | 评估频率 |
| `log_interval` | 日志写入/进度条更新频率 |

#### 优化器与学习率

| 字段 | 作用 |
| --- | --- |
| `learning_rate` / `min_lr` | 优化器 LR 与下限保护 |
| `weight_decay` | AdamW 权重衰减 |
| `warmup_steps` | Cosine 调度预热步数 |

#### MRL（Matryoshka）

| 字段 | 作用 |
| --- | --- |
| `mrl_enabled` | 是否启用多维对齐训练 |
| `mrl_dims` | 参与训练的维度列表（如 128、256） |
| `mrl_dim_weights` | 各维度的损失权重 |

#### 均匀性正则（Spread-out）

| 字段 | 作用 |
| --- | --- |
| `spread_enabled` | 是否启用均匀性正则 |
| `spread_weight` | 正则项权重 |
| `spread_t` | RBF 温度参数 |

#### LoRA 配置

| 字段 | 作用 |
| --- | --- |
| `lora_enabled` | 是否启用 LoRA |
| `lora_r` / `lora_alpha` / `lora_dropout` / `lora_bias` | LoRA 容量/缩放/dropout/bias |
| `lora_task_type` / `lora_target_modules` | 任务类型与注入模块列表 |
| `lora_adapter_path` | 加载已有适配器的路径 |

#### 双向注意力

| 字段 | 作用 |
| --- | --- |
| `bi_attn_enabled` | 启用双向注意力底座（deepseek_v3/qwen3） |

#### 注意力实现方式

| 字段 | 作用 |
| --- | --- |
| `attn_implementation` | 注意力实现（flash/sdpa/eager） |

#### 池化策略

| 字段 | 作用 |
| --- | --- |
| `pooling_mode` | 句向量池化（eos/mean/weighted_mean） |

#### 指令跳过（Skip Instruction）

| 字段 | 作用 |
| --- | --- |
| `skip_instruction` | 是否跳过指令前缀进行池化 |
| `instruction_separator` | 指令分隔符，默认 `Query:` |

### 训练命令

本项目提供了多种训练配置，支持不同的训练策略和优化技术。以下是主要的训练命令：

| 训练命令 | 配置说明 | 功能描述 |
| --- | --- | --- |
| `accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/config.f2llm.json` | 基础配置，完全复现F2LLM项目 | 使用标准F2LLM训练设置，不改变模型结构，采用eos pooling，包含对比学习 |
| `accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/config.mrl_spread.json` | MRL+Spread-out配置 | 启用Matryoshka表示学习和Spread-out均匀性正则，提升模型的多维表示能力和分布均匀性 |
| `accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/config.mrl_spread.bi_mean.json` | 双向注意力+Mean Pooling | 在MRL+Spread-out基础上启用双向注意力和Mean Pooling策略，适合需要全局上下文信息的任务 |
| `accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/config.mrl_spread.lora.bi_mean.json` | LoRA高效微调 | 在双向注意力配置基础上启用LoRA低秩适配，显著减少可训练参数和显存消耗，适合资源受限环境 |

### 配置文件功能详解

1. **config.f2llm.json** - 基础配置：
   - 使用EOS Pooling策略
   - 标准对比学习损失
   - 未启用MRL和Spread-out正则
   - 适合快速验证和基线训练

2. **config.mrl_spread.json** - 增强表示学习：
   - 启用MRL（Matryoshka表示学习）：支持128和256维度的弹性裁剪
   - 启用Spread-out均匀性正则：防止表示空间塌缩，提升向量分布质量
   - 保持EOS Pooling和标准注意力机制

3. **config.mrl_spread.bi_mean.json** - 双向注意力优化：
   - 在MRL+Spread-out基础上启用双向注意力
   - 使用Mean Pooling替代EOS Pooling，更适合全局语义捕捉
   - 适合需要全文上下文理解的任务场景

4. **config.mrl_spread.lora.bi_mean.json** - 高效微调：
   - 在双向注意力配置基础上启用LoRA技术
   - 只在关键投影层（q_proj, v_proj, k_proj等）注入低秩适配器
   - 大幅减少可训练参数（从全量微调降至~1%参数更新）
   - 适合计算资源有限或需要快速迭代的场景

### 训练建议

- **新手入门**：建议从 `config.f2llm.json` 开始，熟悉训练流程后再尝试高级配置
- **性能优化**：`config.mrl_spread.bi_mean.json` 通常能提供最佳的性能表现
- **资源受限**：使用 `config.mrl_spread.lora.bi_mean.json` 进行高效微调
- **实验对比**：可以依次运行不同配置，对比各技术组件对最终效果的影响

## 效果评测（MTEB / T2Retrieval）

MTEB（Massive Text Embedding Benchmark）是当前主流的文本向量评测基准，涵盖检索、语义相似度、分类、聚类、多语言等数十个任务，全面衡量Embedding模型的通用能力。由于MTEB任务众多、运行时间较长，本项目在阶段性验证中仅选用其中的检索任务 T2Retrieval 进行对比评测。

### 基线模型（MTEB已支持，T2Retrieval）

以下结果来自标准MTEB评测流程（抽样 `limits=500` 用于快速验证）：

| 模型 | 任务 | 加载方式 | 执行命令 | nDCG@10 | Recall@10 | Precision@10 |
| --- | --- | --- | --- | --- | --- | --- |
| intfloat.multilingual-e5-large-instruct | T2Retrieval | MTEB内置 | `CUDA_VISIBLE_DEVICES=0 python evaluate_mteb_t2retrieval_models.py --model e5 --device 0 --batch_size 256 --limits 500` | 0.82921 | 0.81743 | 0.41200 |
| BAAI.bge-m3 | T2Retrieval | MTEB内置 | `CUDA_VISIBLE_DEVICES=1 python evaluate_mteb_t2retrieval_models.py --model bge --device 1 --batch_size 256 --limits 500` | 0.81450 | 0.80553 | 0.40676 |
| Qwen3-Embedding-0.6B | T2Retrieval | MTEB内置 | `CUDA_VISIBLE_DEVICES=0 python evaluate_mteb_t2retrieval_models.py --model qwen3embedding --device 0 --batch_size 256 --limits 500` | 0.80492 | 0.79283 | 0.39974 |

结论：三款模型在 T2Retrieval 上的核心指标较为接近。


### MTEB中自定义模型评测

对于MTEB未内置的模型/管线，需要实现自定义模型类再接入评测。为验证评测管线正确性和后训练的效果，我们选取：
- `Qwen/Qwen3-0.6B`：作为基础LLM；
- `Qwen/Qwen3-Embedding-0.6B`：作为正确性对照，使用相同评测脚本跑分。

评测环境均采用 `limits=500` 抽样与批量推理（具体命令与结果路径见参考来源）。

| 模型 | 配置/说明 | 执行命令 | nDCG@10 | MRR@10 | MAP@10 | Recall@100 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-0.6B | 基座 + EOS（指令化，HF模型ID） | `python evaluate_mteb_t2retrieval_models.py --model qwen3eos --batch_size 128 --limits 500 --device 0 --model_id Qwen/Qwen3-0.6B --query_instruction "Instruct: 检索相关段落。\nQuery: "` | 0.01453 | 0.02801 | 0.00711 | 0.08029 |
| Qwen/Qwen3-Embedding-0.6B | 指令化（transformers加载） | `python evaluate_mteb_t2retrieval_models.py --model qwen3eos --batch_size 128 --limits 500 --device 0 --model_id Qwen/Qwen3-Embedding-0.6B --query_instruction "Instruct: 检索相关段落。\nQuery: "` | 0.82824 | 0.91731 | 0.74989 | 0.93618 |

两个模型的作用：
- 使用 `Qwen/Qwen3-Embedding-0.6B` 的结果判断自定义模型类是否正确（对齐到已知强基线）。
- 自定义模型类实现的是 `eos pooling + instruct`，可用于对比 `Qwen/Qwen3-Embedding-0.6B` 相比 `Qwen/Qwen3-0.6B` 的效果提升。

结论：
- 从 `Qwen/Qwen3-Embedding-0.6B` 的结果看出自定义模型类效果正常。
- 从 `Qwen/Qwen3-0.6B` 的结果看出不经过后训练直接用 LLM 的 hidden state 作为表征效果非常差。


### 本项目模型评测（T2Retrieval）

当前已完成对 `config.f2llm.json` 配置的T2Retrieval评测，命令与结果如下：

| 配置 | 评测命令 | 指令模板 | 池化 | checkpoint | nDCG@10 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- |
| config.f2llm.json | `CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_extractor_mteb.py --config configs/config.f2llm.json --tasks T2Retrieval --batch_size 256 --ckpt_dir ./step_122000 --output_folder mteb_results/test8.t2.re4gpu.long.eos.ins.step_122000 --query_instruction "Instruct: 检索相关段落。\nQuery: "` | `Instruct: 检索相关段落。\nQuery: ` | eos | step_122000 | 0.71542 | 0.68685 |

结论：与 SOTA 模型仍有一定差距。可能原因是 T2Retrieval 为中文信息检索任务，而当前训练数据中中文检索任务占比偏低。


其它配置（如 `config.mrl_spread.json`、`config.mrl_spread.bi_mean.json`、`config.mrl_spread.lora.bi_mean.json`）的评测正在进行中，完成后将统一追加到本节表格中以便对比。
