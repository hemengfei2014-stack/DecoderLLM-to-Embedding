import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# 说明（中文）：
# 本文件封装了一个简化的 DecoderLLMEmbeddingExtractor 模型包装器，用于：
# - 从 HuggingFace Transformers 加载指定的预训练语言模型（LLM）；
# - 通过配置传递注意力实现方式（`attn_implementation`，默认 `flash_attention_2`），不再做代码层面的兜底回退；
# - 在前向计算中，仅提取每条输入序列的“最后一个有效 token”的隐状态，作为该序列的语义表示；
# - 返回查询、正例段落以及若干负例段落的末位 token 表示，用于对比学习。
#
# 注意：本包装器不改变 LLM 的结构与注意力机制，仅在前向阶段选择性地取用最后 token 的隐状态（等价于 EOS pooling）。


class DecoderLLMEmbeddingExtractor:
    """DecoderLLMEmbeddingExtractor 模型包装器

    职责：
    - 加载基础 LLM 与对应 tokenizer；
    - 设定精度与设备；
    - 执行前向传播并抽取序列末位 token 的表示。

    属性：
    - lm: 加载的 `AutoModel` 或双向适配模型实例
    - tokenizer: 对应的 `AutoTokenizer`
    - dtype: 默认使用 `torch.bfloat16`
    - device: 由外部训练流程（accelerate.prepare）设置；此处仅占位
    - max_seq_length: 最大序列长度（用于数据预处理阶段限制）

    使用示例：
    >>> from arguments import Args
    >>> args = Args(model_path="Qwen/Qwen3-0.6B", attn_implementation="flash_attention_2")
    >>> m = DecoderLLMEmbeddingExtractor(args.model_path, max_seq_length=1024, args=args)
    >>> # 见 run.py 中的 collate_fn 以构造 batch
    >>> # out = m.forward(batch)
    """
    def __init__(self,
                 model_path,
                 max_seq_length=512,
                 args=None
                 ):

        self.args = args
        self.dtype = torch.bfloat16
        self.device = None  # 设备由外部加速器在 prepare 后设置
        # 注意力实现：由配置传入，默认 `flash_attention_2`，常见取值还包括 `sdpa`、`eager`
        attn_impl = 'flash_attention_2'
        if self.args and hasattr(self.args, 'attn_implementation') and self.args.attn_implementation:
            attn_impl = self.args.attn_implementation
        try:
            if self.args and getattr(self.args, 'bi_attn_enabled', False):
                cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model_type = getattr(cfg, 'model_type', None)
                if model_type not in ['deepseek_v3', 'qwen3']:
                    raise ValueError(f"双向注意力仅支持已适配模型；当前不支持: {model_type}")
                setattr(cfg, '_attn_implementation', attn_impl)
                setattr(cfg, 'attn_implementation', attn_impl)
                if model_type == 'deepseek_v3':
                    from models import DeepseekV3BiModel
                    self.lm = DeepseekV3BiModel.from_pretrained(model_path, config=cfg, trust_remote_code=True, torch_dtype=self.dtype)
                else:
                    from models import Qwen3BiModel
                    self.lm = Qwen3BiModel.from_pretrained(model_path, config=cfg, trust_remote_code=True, torch_dtype=self.dtype)
                attention_impl = getattr(cfg, '_attn_implementation', 'unknown')
                print(f"[DecoderLLMEmbeddingExtractor] Bidirectional Attention enabled: {attention_impl}")
            else:
                self.lm = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=self.dtype, attn_implementation=attn_impl)
                attention_impl = getattr(self.lm.config, 'attn_implementation', 'unknown')
                print(f"[DecoderLLMEmbeddingExtractor] Attention implementation: {attention_impl}")
        except Exception as e:
            raise e
        # 训练时关闭 `use_cache`，避免缓存导致的前向与反向不一致/占用内存
        self.lm.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length

        # LoRA 集成：仅在启用时包裹底座模型；默认 task 为 FEATURE_EXTRACTION
        if self.args and getattr(self.args, 'lora_enabled', False):
            task_type = getattr(TaskType, self.args.lora_task_type) if isinstance(self.args.lora_task_type, str) else self.args.lora_task_type
            target_modules = self.args.lora_target_modules if len(self.args.lora_target_modules) > 0 else ["q_proj", "v_proj"]
            peft_config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                bias=self.args.lora_bias,
                task_type=task_type,
                target_modules=target_modules,
            )
            self.lm = get_peft_model(self.lm, peft_config)
            if hasattr(self.lm, 'enable_input_require_grads'):
                try:
                    self.lm.enable_input_require_grads()
                except Exception as e:
                    raise RuntimeError(f"Critical: Failed to enable input require grads. This is required for LoRA + Gradient Checkpointing training. Error: {e}")
            if getattr(self.args, 'lora_adapter_path', ""):
                try:
                    self.lm.load_adapter(self.args.lora_adapter_path, adapter_name="default")
                except Exception:
                    from peft import PeftModel
                    self.lm = PeftModel.from_pretrained(self.lm, self.args.lora_adapter_path)
            try:
                self.lm.print_trainable_parameters()
            except Exception:
                pass

    def set_device(self):
        """与外部加速器保持一致，记录当前模型所在设备。"""
        self.device = self.lm.device
    
    def forward(self, batch):
        """前向传播：提取每条序列的末位 token 表示

        输入字典（由 `run.py` 的 `collate_fn` 构造）：
        - input_ids: 形状 (N, L) 的张量，N=bs*(2+num_hard_neg)，包含 bs 个 query、bs 个正例 passage、以及 bs*num_hard_neg 个负例
        - attention_mask: 同形状 (N, L)，非 pad 位置为 1
        - seq_lens: 形状 (N,) 的张量，记录每条序列的真实长度（含 EOS），用于索引末位 token
        - bs: 当前批次样本数

        输出字典：
        - query_passage_features: 形状 (bs, 1, H)，H 为隐层维度；每个查询的末位 token 表示
        - passage_passage_features: 形状 (bs, 1, H)；每个正例段落的末位 token 表示
        - negative_passage_features: 若存在负例，形状 (bs, num_hard_neg, H)；否则为 None

        pooling 说明：
        - `eos`（默认）：按 `seq_lens[i]-1` 取末位有效 token；
        - `mean`：对有效长度范围做均值；
        - `weighted_mean`：按位置从 1..L 线性权重做加权均值（示意权重可替换）。

        示例（形状）：
        - bs=2、num_hard_neg=1 时，`input_ids` 共 2(query)+2(正例)+2*1(负例)=6 条序列；
        - `negative_passage_features` 为 (2, 1, H)。
        """
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2*bs) / bs)

        outputs = self.lm(batch['input_ids'], batch['attention_mask'])

        passage_features_all_tokens = outputs.last_hidden_state
        # 依据 `seq_lens[i]-1` 获取每条序列的末位有效 token 的隐状态（batch 维度为 i）
        # 前 bs 条对应 query；中间 bs 条对应正例 passage；其后（若存在）对应负例，按样本分组 reshape
        def _pool(last_hidden, start, end):
            mode = getattr(self.args, 'pooling_mode', 'eos')
            if mode == 'mean':
                if getattr(self.args, 'skip_instruction', False) and 'embed_mask' in batch:
                    outs = []
                    for i in range(start, end):
                        m = batch['embed_mask'][i].bool()
                        outs.append(last_hidden[i, m, :].mean(dim=0))
                    return torch.stack(outs)
                return torch.stack([last_hidden[i, :batch['seq_lens'][i], :].mean(dim=0) for i in range(start, end)])
            elif mode == 'weighted_mean':
                outs = []
                for i in range(start, end):
                    if getattr(self.args, 'skip_instruction', False) and 'embed_mask' in batch:
                        m = batch['embed_mask'][i].bool()
                        L = m.sum().item()
                        if L == 0:
                            outs.append(last_hidden[i, [batch['seq_lens'][i]-1], :].squeeze(0))
                            continue
                        weights = torch.arange(L, device=last_hidden.device) + 1
                        weights = weights / torch.clamp(weights.sum(), min=1e-9)
                        outs.append((last_hidden[i, m, :] * weights.unsqueeze(-1)).sum(dim=0))
                    else:
                        L = batch['seq_lens'][i].item()
                        weights = torch.arange(L, device=last_hidden.device) + 1
                        weights = weights / torch.clamp(weights.sum(), min=1e-9)
                        outs.append((last_hidden[i, :L, :] * weights.unsqueeze(-1)).sum(dim=0))
                return torch.stack(outs)
            else:
                # 默认 eos/last token pooling（右侧 padding）
                if getattr(self.args, 'skip_instruction', False) and 'embed_mask' in batch:
                    outs = []
                    for i in range(start, end):
                        m = batch['embed_mask'][i].bool()
                        idxs = torch.nonzero(m, as_tuple=False).squeeze(-1)
                        if idxs.numel() == 0:
                            outs.append(last_hidden[i, [batch['seq_lens'][i]-1], :].squeeze(0))
                        else:
                            outs.append(last_hidden[i, [idxs[-1]], :].squeeze(0))
                    return torch.stack(outs)
                return torch.stack([last_hidden[i, [batch['seq_lens'][i]-1]] for i in range(start, end)])

        qp = _pool(passage_features_all_tokens, 0, bs)
        pp = _pool(passage_features_all_tokens, bs, 2*bs)
        if qp.dim() == 2:
            qp = qp.unsqueeze(1)
        if pp.dim() == 2:
            pp = pp.unsqueeze(1)
        if num_hard_neg == 0:
            npf = None
        else:
            outs = _pool(passage_features_all_tokens, 2*bs, len(batch['seq_lens']))
            npf = outs.view(bs, num_hard_neg, -1)
        
        return {
            'query_passage_features': qp,
            'passage_passage_features': pp,
            'negative_passage_features': npf
        }
