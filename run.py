"""
DecoderLLMEmbeddingExtractor 训练入口脚本

概览：
- 解析配置（见 `arguments.py`）并初始化 `Accelerator`。
- 从 `args.train_data_path` 读取多个 parquet 数据集，按 0.99/0.01 划分训练/验证。
- 构造 tokenizer，确保有 `pad_token_id`，右侧 padding。
- `collate_fn` 将 query/passage/negatives 组成批次并统一裁剪与 padding。
- `MultiLoader` 以加权随机的方式在多个数据源之间迭代直至全部耗尽。
- 设置优化器与学习率调度器，并通过 `accelerate_train` 执行训练、验证与保存。

说明：脚本不改模型结构；句向量取法为序列末位（通常是 eos）。分类数据集仅使用一个负例；检索/聚类数据集使用 `args.num_hard_neg` 个负例。

示例：
Run（单机多卡）：
torchrun --nproc_per_node 4 \
  run.py --config ./configs/config.expB.mrl_spread.lora.bi_weighted.qwen3.json
"""
from arguments import parse_args
from utils import accelerate_train, CLASSIFICATION_DATASETS
from transformers import (
    AutoTokenizer,
    set_seed,
    get_scheduler
)
import os, json, random
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.state import AcceleratorState
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from model import DecoderLLMEmbeddingExtractor

# 关闭并行分词器的日志/警告，避免多进程时输出干扰
os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = parse_args()
accelerator = Accelerator()
args.num_processes = accelerator.num_processes
accelerator.print(args)

def _stack(input_ids, max_len):
    """将不等长序列裁剪到 `max_len` 后拼成一维张量，再按原长度切回列表。
    这样比逐个 tensor 再 pad 更高效，便于后续统一 padding。
    """
    data = [ids[:max_len] for ids in input_ids]     # input_ids: list of lists
    lens = [len(x) for x in data]
    tensor = torch.tensor(sum(data, []))            # (total_tokens,)
    return tensor.split(lens)                       # list of 1-d tensors


def collate_fn(batch_raw):
    '''
        length of input_ids: bs * (2 + num_hard_neg)
        0 - bs-1: query input ids
        bs - 2*bs-1: passage input ids
        2*bs - 2*bs+num_hard_neg-1: hard neg for sample 1
        2*bs+num_hard_neg*(i-1) - 2*bs+num_hard_neg*i-1: hard neg for sample i (i from 1 to bs)
    '''
    # 分类数据集仅使用 1 个负例；检索/聚类数据集使用配置的 `num_hard_neg`
    num_hard_neg = 1 if batch_raw[0]['dataset_name'] in CLASSIFICATION_DATASETS else args.num_hard_neg
    
    # For classification datasets, only use negative_1
    if batch_raw[0]['dataset_name'] in CLASSIFICATION_DATASETS:
        hard_neg_indices = [0]
    else:
        # 检索/聚类数据集：检查当前样本有哪些 `negative_{i}_input_ids` 可用，并据此采样
        available_negatives = []
        for i in range(24):
            if f'negative_{i+1}_input_ids' in batch_raw[0]:
                available_negatives.append(i)
        
        if len(available_negatives) == 0:
            # Fallback to negative_1 if no negatives found
            hard_neg_indices = [0] if 'negative_1_input_ids' in batch_raw[0] else []
        elif len(available_negatives) < num_hard_neg:
            # Use all available negatives if less than required
            hard_neg_indices = available_negatives
        else:
            # 正常情况：从可用负例中随机采样 `num_hard_neg` 个索引
            hard_neg_indices = random.sample(available_negatives, num_hard_neg)
    
    # Build input_ids list with error checking
    # 逐样本收集负例的 `input_ids`（具备容错：缺失时回退到 negative_1）
    negative_input_ids = []
    for s in batch_raw:
        for i in hard_neg_indices:
            neg_key = f'negative_{i+1}_input_ids'
            if neg_key in s:
                negative_input_ids.append(s[neg_key])
            else:
                # Fallback to negative_1 if specific negative not found
                if 'negative_1_input_ids' in s:
                    negative_input_ids.append(s['negative_1_input_ids'])
    
    if getattr(args, 'skip_instruction', False):
        input_ids = _stack(
            [s['query_input_ids'] for s in batch_raw]+\
            [s['passage_input_ids'] for s in batch_raw]+\
            negative_input_ids,
            args.max_seq_length
        )
        seqlens = torch.tensor([ids.size(0) for ids in input_ids])
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_masks = input_ids.ne(tokenizer.pad_token_id).to(torch.long).contiguous()

        masks = []
        # queries: detect separator from config (default "Query:") and mask only the Query part
        for s in batch_raw:
            q_text = s.get('query', '')
            doc_len = None
            if isinstance(q_text, str):
                sep = getattr(args, 'instruction_separator', '') or 'Query:'
                if sep in q_text:
                    parts = q_text.split(sep, 1)
                    doc = parts[1]
                    ids_doc = tokenizer(doc, add_special_tokens=False, truncation=True, max_length=args.max_seq_length).input_ids
                    doc_len = min(len(ids_doc), args.max_seq_length)
            q_ids_len = min(len(s['query_input_ids']), args.max_seq_length)
            if doc_len is None:
                doc_len = max(q_ids_len - 1, 0)
            prefix_len = q_ids_len - doc_len
            m = [0]*prefix_len + [1]*doc_len
            masks.append(torch.tensor(m))
        # positives: mask all tokens except final eos
        for s in batch_raw:
            p_len = min(len(s['passage_input_ids']), args.max_seq_length)
            m = [1]*max(p_len-1, 0) + ([0] if p_len>0 else [])
            masks.append(torch.tensor(m))
        # negatives: mask all tokens except final eos
        for s in batch_raw:
            for i in hard_neg_indices:
                neg_key = f'negative_{i+1}_input_ids'
                if neg_key in s:
                    n_len = min(len(s[neg_key]), args.max_seq_length)
                    m = [1]*max(n_len-1, 0) + ([0] if n_len>0 else [])
                    masks.append(torch.tensor(m))

        embed_mask = pad_sequence(masks, batch_first=True, padding_value=0)
        batch_dict = {'input_ids': input_ids, 'seq_lens': seqlens, 'attention_mask': attention_masks, 'bs': len(batch_raw), 'dataset_name': batch_raw[0]['dataset_name']}
        batch_dict['embed_mask'] = embed_mask
        return batch_dict
    else:
        # 原路径：不注入指令，直接拼接并右侧 pad
        input_ids = _stack(
            [s['query_input_ids'] for s in batch_raw]+\
            [s['passage_input_ids'] for s in batch_raw]+\
            negative_input_ids,
            args.max_seq_length
        )
        seqlens = torch.tensor([ids.size(0) for ids in input_ids])
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_masks = input_ids.ne(tokenizer.pad_token_id).to(torch.long).contiguous()
        return {'input_ids': input_ids, 'seq_lens': seqlens, 'attention_mask': attention_masks, 'bs': len(batch_raw), 'dataset_name': batch_raw[0]['dataset_name']}


set_seed(0)
if accelerator.is_main_process:
    # 主进程创建输出目录并保存配置，便于复现与审计
    os.makedirs(f"{args.output_dir}", exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:   
        json.dump(args.dict(), f, indent=2)

train_datasets, valid_datasets = [], []
files = sorted([f for f in os.listdir(args.train_data_path) if f.endswith('.parquet')])
if args.max_datasets and args.max_datasets > 0:
    files = files[:args.max_datasets]

for f in files:
    dataset_name = f.split('.parquet')[0]
    ds = load_dataset("parquet", data_files=os.path.join(args.train_data_path, f), cache_dir=args.cache_dir)['train']
    if args.read_rows_limit and args.read_rows_limit > 0:
        # 只读取前 N 行，加快 I/O 与解压速度
        ds = ds.select(range(min(args.read_rows_limit, len(ds))))
    ds = ds.add_column("dataset_name", [dataset_name]*len(ds))
    if args.fast_split:
        # 直接按比例切分索引，避免构建中间数据副本
        n = len(ds)
        n_train = int(n * 0.99)
        train_ds = ds.select(range(n_train))
        valid_ds = ds.select(range(n_train, n))
    else:
        ds = ds.train_test_split(train_size=0.99, shuffle=True, seed=0)
        train_ds = ds['train']
        valid_ds = ds['test']
    if args.train_subset_rows and args.train_subset_rows > 0:
        train_ds = train_ds.select(range(min(args.train_subset_rows, len(train_ds))))
    if args.valid_subset_rows and args.valid_subset_rows > 0:
        valid_ds = valid_ds.select(range(min(args.valid_subset_rows, len(valid_ds))))
    train_datasets.append((dataset_name, train_ds))
    valid_datasets.append((dataset_name, valid_ds))

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
# 确保 tokenizer 有 pad token；若缺失，使用 eos 作为 pad。右侧 padding 便于从末位取句向量
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'

train_loaders = {
    name: DataLoader(ds, shuffle=True, batch_size=args.train_batch_size, drop_last=True, collate_fn=collate_fn)
    for name, ds in train_datasets
}
valid_loaders = {
    name: DataLoader(ds, shuffle=False, batch_size=args.train_batch_size, drop_last=True, collate_fn=collate_fn)
    for name, ds in valid_datasets
}

class MultiLoader:
    """
    Iterates over a dict(name -> DataLoader) and returns complete batches.
    At every __iter__ a new random order is created;
    the epoch ends when every loader is exhausted once.
    """
    def __init__(self, loader_dict):
        self.loader_dict = loader_dict
        for k, v in self.loader_dict.items():
            # 提前对各个 DataLoader 进行 `accelerator.prepare`，以便后续跨设备迭代
            self.loader_dict[k] = accelerator.prepare(v)

    def __len__(self):
        return sum(len(v) for v in self.loader_dict.values())
    
    def reset_epoch(self, epoch):
        self.rng = random.Random(epoch)
        self.iters = {k: iter(v) for k, v in self.loader_dict.items()}
        self.names = list(self.iters.keys())
        self.weights = [len(self.loader_dict[k]) for k in self.names]

    def __iter__(self):
        while self.names:                           # 直到所有 DataLoader 都被耗尽一次
            name = self.rng.choices(self.names, weights=self.weights)[0] # 按剩余样本数加权随机选一个数据源
            try:
                batch = next(self.iters[name])
                yield batch
            except StopIteration:
                idx = self.names.index(name)
                self.names.pop(idx)                 # 该数据源已无 batch
                self.weights.pop(idx)


# determine training steps
override_train_step = False
if args.train_steps < 0:
    # 若未显式指定 train_steps，按 sum(len(loader))*epochs 计算
    args.train_steps = sum(len(v) for v in train_loaders.values()) * args.train_epochs
    override_train_step = True

accelerator.print(f"******************************** Training step before prepare: {args.train_steps} ********************************")
model = DecoderLLMEmbeddingExtractor(args.model_path, args.max_seq_length, args=args)
model.lm.gradient_checkpointing_enable()
try:
    model.lm.enable_input_require_grads()
except Exception:
    pass
# set seed again to make sure that different models share the same seed
set_seed(0)

optimizer = AdamW(model.lm.parameters(),
                  weight_decay=args.weight_decay,
                  lr=args.learning_rate,
                  betas=(0.9, 0.98))

lr_scheduler = get_scheduler("cosine",
                            optimizer=optimizer,
                            num_warmup_steps=args.warmup_steps,
                            num_training_steps=args.train_steps)

try:
    # 若使用 deepspeed，覆盖其 micro-batch 配置为 `args.train_batch_size`
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_batch_size
except Exception:
    pass
model.lm, optimizer, lr_scheduler = accelerator.prepare(
    model.lm, optimizer, lr_scheduler
)
model.set_device()
train_dataloader = MultiLoader(train_loaders)
for k, v in valid_loaders.items():
    valid_loaders[k] = accelerator.prepare(v)

# if training on multiple GPUs, length of dataloader would have changed
if override_train_step:
    # 多 GPU 下，经 `accelerator.prepare` 后 DataLoader 的长度可能变化，需重新计算总步数
    args.train_steps = len(train_dataloader) * args.train_epochs
accelerator.print(f"******************************** Training step after prepare: {args.train_steps} ********************************")


# 统计所有训练数据集的样本总数（用于日志与参考）
total_train_samples = sum(len(ds) for _, ds in train_datasets)
accelerate_train(args, accelerator, model, train_dataloader, valid_loaders,
                 optimizer, lr_scheduler, total_train_samples)
