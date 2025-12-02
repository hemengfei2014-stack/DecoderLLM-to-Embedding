"""
utils.py

职责与结构：
- 训练/验证日志：统一将各损失与学习率写入 TensorBoard，提供分数据集与分任务类型的平均指标；
- 检查点保存：在多卡/分布式环境（Accelerate）下安全保存 tokenizer 与模型权重；
  * 启用 LoRA 时，仅保存适配器权重至 `adapter/` 子目录，便于「基座 + 适配器」部署；
  * 未启用 LoRA 时，保存完整权重；
- 损失计算：
  * in-batch 对比损失（检索类数据集）：跨进程 gather，构造全局候选，做余弦相似 + 温度缩放 + 交叉熵；
  * hard 负例损失：将正样本视为类别 0，若干 hard negatives 作为其他类别，做多类交叉熵；
  * spread 正则（可选）：通过距离均匀性约束避免 collapse；
- 验证流程：逐数据集聚合 hard/in-batch/spread 损失，并统计各任务类型的平均指标；
- 训练主循环：分 epoch、步进迭代，计算损失与反向优化，按间隔记录日志、触发验证与保存 checkpoint。

重要约定：
- 所有损失在任何分支均返回张量（位于正确设备），避免 Python float 与张量混用导致的反向与分布式错误；
- LoRA 保存/加载与训练逻辑解耦，训练过程仅依赖 `model.forward` 提供的末位 token 表示。
"""

from tqdm.auto import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os

CLASSIFICATION_DATASETS = ['amazon_counterfactual', 'amazon_polarity', 'imdb', 'toxic_conversations', 'cola']
CLUSTERING_DATASETS = ['amazon_reviews', 'banking77', 'emotion', 'mtop_intent', 'mtop_domain', 'massive_scenario', 'massive_intent', 'tweet_sentiment_extraction', 'arxiv_clustering_p2p', 'arxiv_clustering_s2s', 'biorxiv_clustering_p2p', 'biorxiv_clustering_s2s', 'medrxiv_clustering_p2p', 'medrxiv_clustering_s2s', 'reddit_clustering_p2p', 'reddit_clustering_s2s', 'stackexchange_clustering_p2p', 'stackexchange_clustering_s2s', 'twentynewsgroups']
RETRIEVAL_DATASETS = ['arguana', 'snli', 'mnli', 'anli', 'paq', 'squad', 'stackexchange', 'msmarco', 'natural_questions', 'hotpotqa', 'fever', 'eli5', 'fiqa', 'bioasq', 'nfcorpus', 'miracl', 'mrtidy', 'scifact', 'qqp', 'stackoverflowdupquestions', 'sts12', 'sts22', 'stsbenchmark', 'amazon_qa', 'cnn_dm', 'coliee', 'paq_part2', 'pubmedqa', 's2orc_abstract_citation', 's2orc_title_abstract', 's2orc_title_citation', 'sentence_compression', 'specter', 'triviaqa', 'xsum', 'stackexchange_part2', 'stackexchangedupquestions_s2s', 'stackexchangedupquestions_p2p']


def write_tensorboard(summary_writer: SummaryWriter, log_dict: dict, completed_steps):
    """将标量日志写入 TensorBoard。
    
    语义：所有值将以当前 `completed_steps` 作为 x 轴写入；
    分布式场景下仅主进程持有有效 `SummaryWriter`，其他进程直接返回。
    """
    if summary_writer is None:
        return
    for key, value in log_dict.items():
        summary_writer.add_scalar(key, value, completed_steps)


def _safe_mean_values(values):
    """安全地计算均值。

    行为：
    - 空输入返回 NaN（`torch.tensor(float('nan')`）；
    - 允许张量/标量混合：张量将 `detach().mean().item()` 后参与平均，避免设备/梯度状态影响；
    - 无法转换为浮点的元素将被忽略。
    """
    values = list(values)
    if len(values) == 0:
        return torch.tensor(float('nan'))
    buf = []
    for v in values:
        if isinstance(v, torch.Tensor):
            # 张量取均值后转为 Python float，避免设备与梯度状态影响日志
            buf.append(v.detach().float().mean().item())
        else:
            try:
                buf.append(float(v))
            except Exception:
                continue
    if len(buf) == 0:
        return torch.tensor(float('nan'))
    return torch.tensor(buf).mean()


def save_checkpoint(args, accelerator, model, output_dir, lr_scheduler):
    accelerator.wait_for_everyone()
    accelerator.print(f"Saving checkpoint to {output_dir}")

    if accelerator.is_main_process:
        model.tokenizer.save_pretrained(output_dir)

    # 支持 LoRA：若启用，保存适配器权重到子目录 adapter；否则保存完整模型
    # 说明：
    # - `accelerator.unwrap_model` 获取未被 DDP/FSDP 包裹的真实模型对象；
    # - 当 LoRA 启用时，适配器权重通过 `unwrapped.save_pretrained(adapter_dir, ...)` 单独持久化；
    # - 失败回退：若 `unwrapped` 为 `PeftModel` 实例，直接调用其 `save_pretrained` 以确保兼容；
    unwrapped = accelerator.unwrap_model(model.lm)
    if getattr(args, 'lora_enabled', False):
        adapter_dir = os.path.join(output_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        try:
            unwrapped.save_pretrained(
                adapter_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model.lm),
            )
        except Exception:
            from peft import PeftModel
            if isinstance(unwrapped, PeftModel):
                unwrapped.save_pretrained(adapter_dir)
            else:
                unwrapped.save_pretrained(
                    output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model.lm),
                )
    else:
        unwrapped.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model.lm),
        )
    accelerator.wait_for_everyone()


def inbatch_loss(
        query_embeddings,  # [bs, d]
        context_embeddings,  # [bs, d]
        criterion,
        accelerator,
        temperature=0.05,
    ):
    """跨进程 in-batch 对比损失（检索类数据集）。

    计算流程：
    - 归一化 `query_embeddings` 与跨进程聚合后的 `context_embeddings`；
    - 计算所有 query 对所有 context 的相似度矩阵并按温度缩放；
    - 标签为本进程样本在全局矩阵中的列偏移（`i + bs * accelerator.process_index`）；
    - 交叉熵约束对应正样本列概率最大；
    - 返回按样本平均后的标量张量。

    示例（形状与标签）：
    - `world_size=2`，每进程 `bs=2`，则 `accelerator.gather(context)` 形状为 `[4, d]`；
    - 进程 0 的 labels 为 `[0, 1]`，进程 1 的 labels 为 `[2, 3]`（按 `bs * process_index` 偏移）；
    - 相似度矩阵 `student_logits` 形状为 `[2, 4]`，每行对齐到对应列标签做交叉熵。
    注意：`accelerator.gather` 的拼接顺序与进程序号一致；确保各进程 `bs` 一致以对齐标签偏移。
    """
    bs = query_embeddings.size(0)
    # L2 归一化，确保相似度为余弦形式
    a_norm = F.normalize(query_embeddings, p=2, dim=-1)

    # gather 当前进程的 context 到所有进程，得到 [bs*world_size, d]
    b_cross_gpus = accelerator.gather(context_embeddings)  # [bs*process, d]
    b_norm_cross_gpus = F.normalize(b_cross_gpus, p=2, dim=-1)

    # 计算 query 对所有 context 的相似度，得到 [bs, bs*world_size]
    student_logits = torch.matmul(a_norm, b_norm_cross_gpus.t()) / temperature

    # 标签为本进程的正样本在全局矩阵中的列索引（按进程偏移）
    labels = torch.arange(bs, device=student_logits.device) + bs * accelerator.process_index
    loss_bs = criterion(student_logits, labels)  # (bs)

    loss = loss_bs.mean()
    return loss

def hard_loss(
        query_embeddings,  # [bs, d]
        context_embeddings,  # [bs, d]
        hard_neg_embeddings,  # [bs, num, d]
        criterion,
        accelerator,
        temperature=0.05,
    ):
    """Hard 负例损失。

    计算流程：
    - 将正样本置于第 0 类，若干负例紧随其后，形成多类分类问题；
    - L2 归一化后计算余弦相似并按温度缩放，使用交叉熵监督类别 0；
    - 若无负例（`hard_neg_embeddings is None`），返回与设备对齐的零张量，保证后续加总与反向稳定。

    示例（形状）：
    - `query_embeddings`: `[bs, d]`
    - `context_embeddings`: `[bs, d]`
    - `hard_neg_embeddings`: `[bs, num_hard, d]`
    - 拼接后得到 `[bs, num_hard+1, d]`，相似度按样本逐类计算，`logits` 形状 `[bs, num_hard+1]`。
    """
    if hard_neg_embeddings is None:
        return torch.tensor(0.0, device=query_embeddings.device)

    bs = query_embeddings.size(0)
    a_norm = F.normalize(query_embeddings, p=2, dim=-1)

    # 拼接正样本在第 0 维，负例紧随其后 => [bs, num_hard+1, d]
    hard_neg_embeddings = torch.concat([
        context_embeddings.unsqueeze(1),
        hard_neg_embeddings
    ], dim=1)

    hard_norm = F.normalize(hard_neg_embeddings, p=2, dim=-1)
    logits = (a_norm.unsqueeze(1) * hard_norm).sum(-1) / temperature  # [bs, num_hard+1]

    # 以 0 类为正样本标签，做交叉熵
    loss_hard = criterion(logits, torch.zeros((bs), dtype=torch.long, device=logits.device)).mean()
    return loss_hard

def uniform_loss(x, t=2.0):
    """均匀性正则。

    目的：鼓励表示在特征空间分布更均匀，降低 collapse 风险；
    实现：对所有样本做 L2 归一化，计算成对距离的 RBF（温度 `t`），取对数均值；
    边界：样本数 < 2 时返回设备对齐的零张量。
    """
    if isinstance(x, list):
        x = torch.concat(x, dim=0)
    x = F.normalize(x, p=2, dim=-1).float()
    if x.size(0) < 2:
        return torch.tensor(0.0, device=x.device)
    d = torch.pdist(x, p=2)
    return torch.log(torch.exp(-t * d.pow(2)).mean())


def validate(args, accelerator, model, valid_loader_dict, criterion, completed_steps, summary_writer):
    """验证流程。

    策略：逐数据集遍历验证 loader，累计 hard/in-batch/spread 损失并做 `accelerator.gather`；
    汇总：对齐维度后做均值统计，额外计算各任务类型（检索/分类/聚类）的宏平均指标；
    过程：评估阶段禁用梯度，全程 `torch.no_grad()`。

    说明：
    - 为兼容多卡，单步损失先 `accelerator.gather` 成同形张量后再做均值；
    - MRL 多维场景下，按维度与权重在 `q[:, :m]` 与 `p[:, :m]` 上分别计算并加权；
    - Spread 正则仅参与日志与总损失，不影响标签构造。
    """
    eval_log_dict = {}
    for dataset_name, valid_dataloader in valid_loader_dict.items():
        loss_ls, loss_hard_ls, loss_spread_ls = [], [], []
        for batch in valid_dataloader:
            with torch.no_grad():
                outputs = model.forward(batch)
                q = outputs['query_passage_features'].squeeze(1)
                p = outputs['passage_passage_features'].squeeze(1)
                n = outputs['negative_passage_features']
                if args.mrl_enabled and len(args.mrl_dims) > 0:
                    w = args.mrl_dim_weights if len(args.mrl_dim_weights) == len(args.mrl_dims) else [1.0] * len(args.mrl_dims)
                    loss_hard = 0.0
                    for i, m in enumerate(args.mrl_dims):
                        loss_hard = loss_hard + w[i] * hard_loss(q[:, :m], p[:, :m], None if n is None else n[:, :, :m], criterion, accelerator)
                else:
                    loss_hard = hard_loss(q, p, n, criterion, accelerator)
                # gather 单个/标量张量到所有进程便于统一统计
                loss_hard_ls.append(accelerator.gather(loss_hard).float())
                if dataset_name in RETRIEVAL_DATASETS:
                    if args.mrl_enabled and len(args.mrl_dims) > 0:
                        w = args.mrl_dim_weights if len(args.mrl_dim_weights) == len(args.mrl_dims) else [1.0] * len(args.mrl_dims)
                        loss = 0.0
                        for i, m in enumerate(args.mrl_dims):
                            loss = loss + w[i] * inbatch_loss(q[:, :m], p[:, :m], criterion, accelerator)
                    else:
                        loss = inbatch_loss(q, p, criterion, accelerator)
                    loss_ls.append(accelerator.gather(loss).float())
                if args.spread_enabled and args.spread_weight > 0.0:
                    u = uniform_loss([q, p], t=args.spread_t)
                    loss_spread_ls.append(accelerator.gather(u).float())

        accelerator.wait_for_everyone()
        if len(loss_hard_ls) > 0:
            loss_hard_ls = torch.stack([t.view(1) if t.dim() == 0 else t for t in loss_hard_ls]).squeeze(-1)
            eval_log_dict[f'{dataset_name}/valid_loss_hard'] = loss_hard_ls.mean()
        if dataset_name in RETRIEVAL_DATASETS and len(loss_ls) > 0:
            loss_ls = torch.stack([t.view(1) if t.dim() == 0 else t for t in loss_ls]).squeeze(-1)
            eval_log_dict[f"{dataset_name}/valid_loss_in_batch"] = loss_ls.mean()
        if len(loss_spread_ls) > 0:
            loss_spread_ls = torch.stack([t.view(1) if t.dim() == 0 else t for t in loss_spread_ls]).squeeze(-1)
            eval_log_dict[f'{dataset_name}/valid_loss_spread'] = loss_spread_ls.mean()

    def _safe_mean(values):
        if len(values) == 0:
            return torch.tensor(float('nan'))
        return torch.stack(values).mean()

    # 分类型统计平均验证损失，便于宏观比较
    eval_log_dict['Avg/retrieval/valid_loss_in_batch'] = _safe_mean([
        v for k, v in eval_log_dict.items()
        if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('valid_loss_in_batch')
    ])
    eval_log_dict['Avg/retrieval/valid_loss_hard'] = _safe_mean([
        v for k, v in eval_log_dict.items()
        if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('valid_loss_hard')
    ])
    eval_log_dict['Avg/classification/valid_loss_hard'] = _safe_mean([
        v for k, v in eval_log_dict.items()
        if k.split('/')[0] in CLASSIFICATION_DATASETS and k.endswith('valid_loss_hard')
    ])
    eval_log_dict['Avg/clustering/valid_loss_hard'] = _safe_mean([
        v for k, v in eval_log_dict.items()
        if k.split('/')[0] in CLUSTERING_DATASETS and k.endswith('valid_loss_hard')
    ])
    eval_log_dict['Avg/all/valid_loss_spread'] = _safe_mean([
        v for k, v in eval_log_dict.items()
        if k.endswith('valid_loss_spread')
    ])

    if accelerator.is_main_process:
        write_tensorboard(summary_writer, eval_log_dict, completed_steps)
    accelerator.print(f"[Validation] Step = {completed_steps}")
        

def accelerate_train(args,
                     accelerator,
                     model,
                     train_dataloader,
                     valid_loader_dict,
                     optimizer,
                     lr_scheduler,
                     num_train_samples):
    """训练主循环。

    工作流：
    - 训练信息打印与参与数据集筛选；
    - 初始化日志器、损失函数（逐样本返回，便于后续平均）与缓冲；
    - 迭代各 epoch 与 batch：
      * 前向得到 query/pass/neg 表示；
      * 计算 hard/in-batch/spread（按数据集/开关条件）；
      * 统一将所有损失转为同设备张量并加总为 `loss_total`；
      * 反向 + 优化 + 调度，并做最小学习率保护；
      * 固定步长写训练日志（含分数据集与宏平均指标）；
      * 触发验证与按步保存 `step_*` checkpoint；
    - 每个 epoch 收尾保存一次 `epoch_*` checkpoint 并补做验证。

    细节：
    - `CrossEntropyLoss(reduction='none')` 返回逐样本损失，便于按样本数做均值；
    - 为避免 `float` 与张量混用导致的反向错误，所有分支均返回张量并在加总前与设备对齐；
    - `accelerator.backward` 兼容多卡；`optimizer.zero_grad()` 每步清梯度；
    - 通过 `min_lr` 做调度下限保护，避免过小学习率引发数值问题。
    """
    accelerator.print("**************************************** Start training ****************************************")
    accelerator.print(f" Num train samples = {num_train_samples}")
    accelerator.print(f" Num epochs = {args.train_epochs}")
    accelerator.print(f" Per device batch size = {args.train_batch_size}")
    accelerator.print(f" Global batch size = {args.train_batch_size * accelerator.num_processes}")
    accelerator.print(f" Step per epoch = {len(train_dataloader)}")
    accelerator.print(f" Total training steps = {args.train_steps}")
    accelerator.print("************************************************************************************************")

    # 仅保留本次训练实际存在的子数据集，用于日志聚合
    global RETRIEVAL_DATASETS, CLASSIFICATION_DATASETS, CLUSTERING_DATASETS
    RETRIEVAL_DATASETS = [ds for ds in RETRIEVAL_DATASETS if ds in train_dataloader.loader_dict.keys()]
    CLASSIFICATION_DATASETS = [ds for ds in CLASSIFICATION_DATASETS if ds in train_dataloader.loader_dict.keys()]
    CLUSTERING_DATASETS = [ds for ds in CLUSTERING_DATASETS if ds in train_dataloader.loader_dict.keys()]

    summary_writer = SummaryWriter(log_dir=args.tb_dir) if accelerator.is_main_process else None
    criterion = CrossEntropyLoss(reduction='none')  # 返回逐样本损失，便于后续平均
    pbar = tqdm(range(args.train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    # 为分数据集统计损失与样本计数，便于稳定写日志
    loss_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
    loss_hard_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}
    count_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
    count_hard_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}

    model.lm.train()
    for epoch in range(args.train_epochs):
        accelerator.print(f"*************** Starting epoch {epoch+1} ***************")
        train_dataloader.reset_epoch(epoch)
        for batch in train_dataloader:
            outputs = model.forward(batch)
            q = outputs['query_passage_features'].squeeze(1)
            p = outputs['passage_passage_features'].squeeze(1)
            n = outputs['negative_passage_features']
            if args.mrl_enabled and len(args.mrl_dims) > 0:
                w = args.mrl_dim_weights if len(args.mrl_dim_weights) == len(args.mrl_dims) else [1.0] * len(args.mrl_dims)
                loss_hard = 0.0
                for i, m in enumerate(args.mrl_dims):
                    loss_hard = loss_hard + w[i] * hard_loss(q[:, :m], p[:, :m], None if n is None else n[:, :, :m], criterion, accelerator)
            else:
                loss_hard = hard_loss(q, p, n, criterion, accelerator)
            dataset_name = batch['dataset_name']
            count_hard_dict[dataset_name] += 1
            loss_hard_dict[dataset_name] += loss_hard.detach().float()

            # in-batch 检索损失（仅在检索类数据集上计算）
            if dataset_name in RETRIEVAL_DATASETS:
                if args.mrl_enabled and len(args.mrl_dims) > 0:
                    w = args.mrl_dim_weights if len(args.mrl_dim_weights) == len(args.mrl_dims) else [1.0] * len(args.mrl_dims)
                    loss = 0.0
                    for i, m in enumerate(args.mrl_dims):
                        loss = loss + w[i] * inbatch_loss(q[:, :m], p[:, :m], criterion, accelerator)
                else:
                    loss = inbatch_loss(q, p, criterion, accelerator)
                count_dict[dataset_name] += 1
                loss_dict[dataset_name] += loss.detach().float()
            else:
                loss = torch.tensor(0.0, device=model.lm.device)

            loss_spread = torch.tensor(0.0, device=model.lm.device)
            if args.spread_enabled and args.spread_weight > 0.0:
                if args.mrl_enabled and len(args.mrl_dims) > 0:
                    w = args.mrl_dim_weights if len(args.mrl_dim_weights) == len(args.mrl_dims) else [1.0] * len(args.mrl_dims)
                    us = torch.tensor(0.0, device=model.lm.device)
                    for i, m in enumerate(args.mrl_dims):
                        us = us + w[i] * uniform_loss([q[:, :m], p[:, :m]], t=args.spread_t)
                    loss_spread = args.spread_weight * us
                else:
                    loss_spread = args.spread_weight * uniform_loss([q, p], t=args.spread_t)

            loss_total = (
                loss if isinstance(loss, torch.Tensor) else torch.tensor(loss, device=model.lm.device)
            ) + (
                loss_hard if isinstance(loss_hard, torch.Tensor) else torch.tensor(loss_hard, device=model.lm.device)
            ) + loss_spread

            if not loss_total.requires_grad:
                loss_total = loss_total + (q.sum() + p.sum()) * 0.0

            # 反向、优化与调度
            accelerator.backward(loss_total)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # 学习率下限保护，避免过小导致数值不稳定
            if optimizer.param_groups[0]['lr'] < args.min_lr:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = args.min_lr

            # 训练日志：按固定步长聚合与写入
            completed_steps += 1
            if completed_steps % args.log_interval == 0:
                pbar.update(args.log_interval)

                train_log_dict = {"lr": optimizer.param_groups[0]['lr']}
                # 检索类数据集的 in-batch 损失（按样本计数平均）
                for k in loss_dict.keys():
                    count = accelerator.gather(count_dict[k]).sum()
                    if count > 0:
                        train_log_dict[f"{k}/training_loss_in_batch"] = accelerator.gather(loss_dict[k]).sum() / count
                # 全部数据集的 hard 负例损失（按样本计数平均）
                for k in loss_hard_dict.keys():
                    count = accelerator.gather(count_hard_dict[k]).sum()
                    if count > 0:
                        train_log_dict[f"{k}/training_loss_hard"] = accelerator.gather(loss_hard_dict[k]).sum() / count
                if args.spread_enabled and args.spread_weight > 0.0:
                    train_log_dict['Avg/all/training_loss_spread'] = _safe_mean_values([
                        loss_spread
                    ])
                # 分类型的平均训练损失，便于宏观追踪
                train_log_dict['Avg/retrieval/training_loss_in_batch'] = _safe_mean_values([
                    v for k, v in train_log_dict.items()
                    if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('training_loss_in_batch')
                ])
                train_log_dict['Avg/retrieval/training_loss_hard'] = _safe_mean_values([
                    v for k, v in train_log_dict.items()
                    if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('training_loss_hard')
                ])
                train_log_dict['Avg/classification/training_loss_hard'] = _safe_mean_values([
                    v for k, v in train_log_dict.items()
                    if k.split('/')[0] in CLASSIFICATION_DATASETS and k.endswith('training_loss_hard')
                ])
                train_log_dict['Avg/clustering/training_loss_hard'] = _safe_mean_values([
                    v for k, v in train_log_dict.items()
                    if k.split('/')[0] in CLUSTERING_DATASETS and k.endswith('training_loss_hard')
                ])

                accelerator.print(f"[Train] Step = {completed_steps}")
                if accelerator.is_main_process:
                    write_tensorboard(summary_writer, train_log_dict, completed_steps)
                # 清空缓冲，下一段日志重新累计
                loss_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
                loss_hard_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}
                count_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
                count_hard_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}

            # 到达验证步则进行评估
            if completed_steps % args.validation_steps == 0:
                model.lm.eval()
                validate(args, accelerator, model, valid_loader_dict, criterion, completed_steps, summary_writer)
                model.lm.train()

            # 到达保存步则保存 checkpoint（step_*）
            if args.checkpointing_steps and completed_steps % args.checkpointing_steps == 0:
                output_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                save_checkpoint(args, accelerator, model, output_dir, lr_scheduler)

            if completed_steps >= args.train_steps:
                break

        # 每个 epoch 结尾保存一次 checkpoint，并补做验证
        output_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        save_checkpoint(args, accelerator, model, output_dir, lr_scheduler)
        if completed_steps % args.validation_steps != 0:
            model.lm.eval()
            validate(args, accelerator, model, valid_loader_dict, criterion, completed_steps, summary_writer)
            model.lm.train()

    if summary_writer:
        summary_writer.close()
