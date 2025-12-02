from dataclasses import dataclass, asdict
import argparse, json

"""
arguments.py

用途：集中管理训练/验证的配置项，并支持从 JSON 配置文件加载。

约定：
- 仅在 `parse_args()` 中读取 `--config`，其余参数从 JSON 文件中解析到 `Args`。
- 字段分组见 `Args` 的类注释；默认值用于简化入参与便于调试。

示例：
{
  "model_path": "Qwen/Qwen3-0.6B",
  "experiment_id": "expB.mrl_spread.lora.bi_weighted.qwen3",
  "output_dir": "./outputs",
  "tb_dir": "./tensorboard",
  "cache_dir": "~/.cache/huggingface",
  "train_data_path": "./training_data/parquet",
  "train_batch_size": 8,
  "max_seq_length": 2048,
  "learning_rate": 1e-4,
  "num_hard_neg": 7,
  "lora_enabled": true,
  "bi_attn_enabled": false,
  "pooling_mode": "eos",
  "attn_implementation": "flash_attention_2"
}
"""


@dataclass
class Args:
    """DecoderLLMEmbeddingExtractor 训练配置

    目录：
    - 基本路径：`model_path`、`output_dir`、`tb_dir`、`cache_dir`、`train_data_path`
    - 训练超参：`train_batch_size`、`max_seq_length`、`learning_rate`、`min_lr`、`weight_decay`、`warmup_steps`
    - 数据采样：`train_subset_rows`、`valid_subset_rows`、`max_datasets`、`fast_split`、`read_rows_limit`
    - 训练步数/轮次：`train_steps`（优先生效，<0 时按轮次计算）、`train_epochs`
    - 日志与评估：`log_interval`、`checkpointing_steps`、`validation_steps`
    - LoRA：`lora_enabled`、`lora_r`、`lora_alpha`、`lora_dropout`、`lora_bias`、`lora_task_type`、`lora_target_modules`、`lora_adapter_path`
    - 表示学习：`num_hard_neg`、MRL（`mrl_enabled`、`mrl_dims`、`mrl_dim_weights`）、Spread（`spread_enabled`、`spread_weight`、`spread_t`）
    - 双向注意力与池化：`bi_attn_enabled`、`pooling_mode`、`attn_implementation`
    """

    model_path: str
    experiment_id: str
    # save dir
    output_dir: str
    tb_dir: str
    cache_dir: str
    # training arguments
    train_data_path: str
    train_batch_size: int = 8
    max_seq_length: int = 2048
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    warmup_steps: int = 100
    # embedding-related settings
    num_hard_neg: int = 7
    mrl_enabled: bool = False
    mrl_dims: list = None
    mrl_dim_weights: list = None
    spread_enabled: bool = False
    spread_weight: float = 0.0
    spread_t: float = 2.0
    train_subset_rows: int = 0
    valid_subset_rows: int = 0
    max_datasets: int = 0
    fast_split: bool = True
    read_rows_limit: int = 0
    # train steps take precedence over epochs, set to -1 to disable
    train_steps: int = -1
    train_epochs: int = 5
    log_interval: int = 20
    checkpointing_steps: int = 100
    validation_steps: int = 100
    # just placeholder, for logging purpose
    num_processes: int=0
    # lora
    lora_enabled: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_task_type: str = "FEATURE_EXTRACTION"
    lora_target_modules: list = None
    lora_adapter_path: str = ""
    # bidirectional attention & pooling
    bi_attn_enabled: bool = False
    pooling_mode: str = "eos"  # options: eos | mean | weighted_mean
    attn_implementation: str = "flash_attention_2"
    skip_instruction: bool = False
    instruction_separator: str = ""

    def dict(self):
        return asdict(self)


def parse_args():
    """解析命令行配置并构造 `Args`。

    行为：
    - 仅支持 `--config path/to.json`；
    - 自动填充缺省列表字段，避免 `None` 导致后续拼接/迭代错误；
    - 按 `experiment_id` 拼接最终 `output_dir` 与 `tb_dir`，便于分实验管理。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    arg = parser.parse_args()
    with open(arg.config) as f:
        config = json.load(f)
    args = Args(**config)
    if args.mrl_dims is None:
        args.mrl_dims = []
    if args.mrl_dim_weights is None:
        args.mrl_dim_weights = []
    if args.lora_target_modules is None:
        args.lora_target_modules = []
    args.output_dir = f"{args.output_dir}/{args.experiment_id}"
    args.tb_dir = f"{args.tb_dir}/{args.experiment_id}"
    return args
