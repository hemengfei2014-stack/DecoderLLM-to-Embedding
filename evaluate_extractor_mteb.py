import argparse
import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer
from mteb import MTEB
from model import DecoderLLMEmbeddingExtractor


class DecoderLLMEmbeddingExtractorEncoder:
    def __init__(self, args, ckpt_dir=None, query_instruction=None):
        self.args = args
        self.query_instruction = query_instruction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_dir = args.model_path
        if ckpt_dir and os.path.isdir(ckpt_dir):
            adapter_path = os.path.join(ckpt_dir, "adapter")
            if os.path.isdir(adapter_path):
                self.args.lora_enabled = True
                self.args.lora_adapter_path = adapter_path
            else:
                load_dir = ckpt_dir
        self.model = DecoderLLMEmbeddingExtractor(load_dir, max_seq_length=args.max_seq_length, args=args)
        self.model.lm.to(self.device)
        self.model.lm.eval()
        self._use_dp = torch.cuda.is_available() and torch.cuda.device_count() > 1
        self._dp_model = torch.nn.DataParallel(self.model.lm, device_ids=list(range(torch.cuda.device_count()))) if self._use_dp else None
        if self._dp_model is not None:
            self._dp_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

    def _tokenize(self, texts, is_query: bool = False):
        if is_query and self.query_instruction:
            texts = [self.query_instruction + t for t in texts]
        ids = []
        seqlens = []
        masks = []
        sep = getattr(self.args, "instruction_separator", "Query:")
        for t in texts:
            if isinstance(t, str):
                tok = self.tokenizer(t, add_special_tokens=False, truncation=True, max_length=self.args.max_seq_length)
                input_ids = tok.input_ids + [self.tokenizer.eos_token_id]
            else:
                input_ids = [self.tokenizer.eos_token_id]
            if getattr(self.args, "skip_instruction", False) and isinstance(t, str):
                doc_len = None
                if is_query:
                    if sep in t:
                        parts = t.split(sep, 1)
                        doc = parts[1]
                        ids_doc = self.tokenizer(doc, add_special_tokens=False, truncation=True, max_length=self.args.max_seq_length).input_ids
                        doc_len = min(len(ids_doc), self.args.max_seq_length)
                    q_len = min(len(input_ids), self.args.max_seq_length)
                    if doc_len is None:
                        doc_len = max(q_len - 1, 0)
                    prefix_len = q_len - doc_len
                    m = [0] * prefix_len + [1] * doc_len
                else:
                    q_len = min(len(input_ids), self.args.max_seq_length)
                    m = [1] * max(q_len - 1, 0) + ([0] if q_len > 0 else [])
                masks.append(torch.tensor(m, dtype=torch.long))
            ids.append(torch.tensor(input_ids[: self.args.max_seq_length], dtype=torch.long))
            seqlens.append(len(ids[-1]))
        return ids, torch.tensor(seqlens, dtype=torch.long), masks

    def _pad(self, ids):
        return torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def _pool_batch(self, last_hidden, seqlens, embed_mask=None):
        mode = getattr(self.args, "pooling_mode", "eos")
        outs = []
        for i in range(last_hidden.size(0)):
            L = int(seqlens[i].item())
            if embed_mask is not None and i < embed_mask.size(0):
                m = embed_mask[i]
                dim_len = last_hidden.size(1)
                if m.size(0) < dim_len:
                    m = torch.nn.functional.pad(m, (0, dim_len - m.size(0)), value=0)
                m = m[:dim_len].bool()
                if mode == "mean":
                    if m.sum().item() == 0:
                        outs.append(last_hidden[i, L - 1, :])
                    else:
                        outs.append(last_hidden[i, m, :].mean(dim=0))
                elif mode == "weighted_mean":
                    if m.sum().item() == 0:
                        outs.append(last_hidden[i, L - 1, :])
                    else:
                        w = torch.arange(m.sum().item(), device=last_hidden.device) + 1
                        w = w / torch.clamp(w.sum(), min=1e-9)
                        outs.append((last_hidden[i, m, :] * w.unsqueeze(-1)).sum(dim=0))
                else:
                    idxs = torch.nonzero(m, as_tuple=False).squeeze(-1)
                    if idxs.numel() == 0:
                        outs.append(last_hidden[i, L - 1, :])
                    else:
                        outs.append(last_hidden[i, idxs[-1], :])
            else:
                if mode == "mean":
                    outs.append(last_hidden[i, :L, :].mean(dim=0))
                elif mode == "weighted_mean":
                    w = torch.arange(L, device=last_hidden.device) + 1
                    w = w / torch.clamp(w.sum(), min=1e-9)
                    outs.append((last_hidden[i, :L, :] * w.unsqueeze(-1)).sum(dim=0))
                else:
                    outs.append(last_hidden[i, L - 1, :])
        return torch.stack(outs)

    def _encode_impl(self, texts, batch_size, is_query: bool = False):
        res = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                ids, seqlens, masks = self._tokenize(batch, is_query=is_query)
                input_ids = self._pad(ids).to(self.device)
                L = input_ids.size(1)
                attention_mask = (torch.arange(L, device=self.device).unsqueeze(0) < seqlens.to(self.device).unsqueeze(1)).to(torch.long)
                embed_mask = None
                if getattr(self.args, "skip_instruction", False) and len(masks) > 0:
                    embed_mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0).to(self.device)
                m = self._dp_model if self._dp_model is not None else self.model.lm
                outputs = m(input_ids, attention_mask)
                last_hidden = outputs.last_hidden_state
                pooled = self._pool_batch(last_hidden, seqlens.to(self.device), embed_mask)
                res.append(pooled.detach().to(torch.float32).cpu())
        if len(res) == 0:
            return []
        x = torch.cat(res, dim=0)
        x = torch.nn.functional.normalize(x, p=2, dim=-1).to(torch.float32)
        return [t.numpy() for t in x]

    def encode(self, sentences, batch_size=32, **kwargs):
        return self._encode_impl(sentences, batch_size, is_query=False)

    def encode_queries(self, queries, batch_size=32, **kwargs):
        return self._encode_impl(queries, batch_size, is_query=True)

    def encode_corpus(self, corpus, batch_size=32, **kwargs):
        texts = []
        for c in corpus:
            if isinstance(c, dict):
                if "text" in c and "title" in c and c["title"]:
                    texts.append(f"{c['title']} {c['text']}")
                elif "text" in c:
                    texts.append(c["text"])
                else:
                    texts.append("")
            else:
                texts.append(str(c))
        return self._encode_impl(texts, batch_size, is_query=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tasks", type=str, nargs="*", default=["T2Retrieval"]) 
    p.add_argument("--task_langs", type=str, nargs="*", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output_folder", type=str, default=None)
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--query_instruction", type=str, default=None)
    a = p.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    from arguments import Args
    args = Args(**cfg)
    if args.mrl_dims is None:
        args.mrl_dims = []
    if args.mrl_dim_weights is None:
        args.mrl_dim_weights = []
    if args.lora_target_modules is None:
        args.lora_target_modules = []
    model = DecoderLLMEmbeddingExtractorEncoder(args, ckpt_dir=a.ckpt_dir, query_instruction=a.query_instruction)
    exp_id = cfg.get("experiment_id", "exp")
    out_dir = a.output_folder or os.path.join("mteb_results", exp_id)
    os.makedirs(out_dir, exist_ok=True)
    evaluation = MTEB(tasks=a.tasks if a.tasks else None, task_langs=a.task_langs)
    evaluation.run(model, output_folder=out_dir, encode_kwargs={"batch_size": a.batch_size})


if __name__ == "__main__":
    main()
