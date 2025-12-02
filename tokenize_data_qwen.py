"""
功能概述：
- 将原始训练数据集（parquet）中的文本字段（query、passage、negative_i）用 Qwen3-0.6B 的分词器进行分词；
- 每个序列末尾强制追加一个 eos token，保证句向量可用（与模型前向中使用末位/eos 表征一致）；
- 为 query 直接分词；为 passage 和所有 negative 聚合去重后统一分词，再按文本映射回每条样本，避免重复分词带来的开销；
- 通过多进程并行加速分词；
- 输出到 `data_tokenized_qwen/{ds_name}`（需要确保该目录存在）。

关键设计：
- `max_seq_length = 1023`：由于我们会在末尾追加一个 eos，因此总长度不超过 1024（常见上下文长度限制）。
- `add_special_tokens=False`：不额外注入 BOS/SEP 等特殊符号，保持干净的文本序列，由我们手动追加 eos。
- 分类数据集只有 1 个负例（`negative_1`），检索/聚类数据集有 24 个负例（`negative_1`~`negative_24`）。此处用是否存在 `negative_2` 来进行简易判断。
"""
from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer
from tqdm.auto import tqdm


# 初始化分词器（本地模型路径），如果使用其他模型，请相应修改路径
tokenizer = AutoTokenizer.from_pretrained('models/qwen3-0.6b')
# 设为 1023，以便在追加 eos 后不超过 1024 的常见上限
max_seq_length = 1023


def process_sent(sentence):
    """对单句进行分词：
    - 关闭 `add_special_tokens`，避免自动添加 BOS/SEP 等符号；
    - 打开 `truncation` 并限制 `max_length=1023`；
    - 手动在末尾追加一个 `eos_token_id`；
    返回：numpy 数组形式的 token id 序列。
    """
    # 确保每个序列末尾都有 eos token
    tokenizer_outputs = tokenizer(
        sentence,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False,
    )

    return np.array(tokenizer_outputs.input_ids + [tokenizer.eos_token_id])


def process_sent_batch(s):
    """对 Pandas Series 批量分词（逐元素 apply）。"""
    return s.apply(process_sent)

def parallelize(data, func, num_of_processes=8):
    """使用多进程对数据进行并行处理：
    - 将索引均分为 `num_of_processes` 份；
    - 通过 `Pool.map` 并行执行 `func`；
    - 拼接分块处理后的结果并返回。
    注意：并行度需与机器 CPU 核心数/负载相匹配，过高会导致竞争或内存压力。
    """
    indices = np.array_split(data.index, num_of_processes)
    data_split = [data.iloc[idx] for idx in indices]
    with Pool(num_of_processes) as pool:
        data = pd.concat(pool.map(func, data_split))
    return data


# 原始训练数据目录（每个 parquet 文件对应一个数据集）
root_dir = 'training_data'
for ds_name in tqdm(sorted(os.listdir(root_dir))):
    print(ds_name, flush=True)

    # 读取该数据集的 parquet 文件，期望包含：query、passage、negative_i（若为检索/聚类）
    df = pd.read_parquet(f"{root_dir}/{ds_name}")
    # 对 query 字段直接并行分词；这里并行进程数设置为 62（与原始代码一致），可按实际机器调整
    df['query_input_ids'] = parallelize(df['query'], process_sent_batch, 62)

    # 判断负例数量：如果存在 `negative_2`，则视为 24 个负例；否则仅 1 个负例（分类数据集）
    num_neg = 24 if 'negative_2' in df.keys() else 1

    # 构建需要分词的文本集合：包含 passage 以及所有 negative 文本
    ls = df.passage.to_list()
    for i in range(1, num_neg+1):
        ls += df[f'negative_{i}'].to_list()
    # 去重以避免对相同文本重复分词（显著降低计算量）
    ls = list(set(ls))
    df_tmp = pd.DataFrame({'text': ls})
    # 并行分词（对去重后的文本集），生成 `input_ids`
    df_tmp['input_ids'] = parallelize(df_tmp['text'], process_sent_batch, 62)
    # 设为按文本查找的索引，便于后续按原始样本映射回 token 序列
    df_tmp = df_tmp.set_index('text')

    # 为每条样本映射出 passage 的分词结果
    df['passage_input_ids'] = df.passage.map(df_tmp.input_ids)

    # 为每条样本映射出各个 negative 的分词结果
    for i in range(1, num_neg+1):
        df[f'negative_{i}_input_ids'] = df[f'negative_{i}'].map(df_tmp.input_ids)

    # 写回到 `data_tokenized_qwen/{ds_name}`（请确保该目录已存在）
    df.to_parquet(f'data_tokenized_qwen/{ds_name}', index=False)
