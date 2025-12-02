import argparse
import os
import re
import json
import mteb
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


def compute_out_dir_qwen3eos(model_name: str, output_folder: str, query_instruction: str | None):
    is_local_ckpt = False
    step_match = re.search(r"step_(\d+)", str(model_name).strip())
    if os.path.isdir(str(model_name)) and step_match is not None:
        is_local_ckpt = True
    ins = ".INS" if (query_instruction is not None and len(str(query_instruction).strip()) > 0) else ""
    if is_local_ckpt:
        cfg_path = os.path.join(str(model_name), "config.json")
        arch = None
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            archs = cfg.get("architectures", [])
            arch = archs[0] if isinstance(archs, list) and len(archs) > 0 else None
        except Exception:
            arch = None
        step_val = step_match.group(0) if step_match else "step_0"
        base_name = f"{arch}.{step_val}.FT{ins}.HF" if arch else f"Unknown.{step_val}.FT{ins}.HF"
        return os.path.join(output_folder, base_name)
    base_name = str(model_name).replace("/", ".")
    suffix = ".EMB" if ("embedding" in str(model_name).lower()) else ".BASE"
    return os.path.join(output_folder, f"{base_name}{suffix}{ins}.HF")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--output_folder", type=str, default="mteb_results")
    p.add_argument("--limits", type=int, default=None)
    p.add_argument("--model", type=str, choices=["e5", "bge", "qwen3eos", "qwen3embedding"], default="e5")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--query_instruction", type=str, default=None)
    p.add_argument("--model_id", type=str, default=None)
    a = p.parse_args()

    if a.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.device

    tasks = mteb.get_tasks(tasks=["T2Retrieval"])
    os.makedirs(a.output_folder, exist_ok=True)

    if a.model == "e5":
        model_name = "intfloat/multilingual-e5-large-instruct"
        model = mteb.get_model(model_name)
        out_dir = os.path.join(a.output_folder, "intfloat.multilingual-e5-large-instruct")
    elif a.model == "bge":
        model_name = "BAAI/bge-m3"
        model = mteb.get_model(model_name)
        out_dir = os.path.join(a.output_folder, "BAAI.bge-m3")
    elif a.model == "qwen3embedding":
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        model = mteb.get_model(model_name)
        out_dir = os.path.join(a.output_folder, "Qwen.Qwen3-Embedding-0.6B")
    elif a.model == "qwen3eos":
        model_name = a.model_id or "Qwen/Qwen3-0.6B"
        class Qwen3EOSWrapper:
            def __init__(self, model_id: str, device: str = "cuda", max_length: int = 512, dtype: str = "bf16", query_instruction: str | None = None):
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.tokenizer.padding_side = "left"
                torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16 if dtype == "fp16" else torch.float32
                self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=torch_dtype)
                self.model.eval()
                self.device = torch.device(device if torch.cuda.is_available() and device is not None else "cpu")
                self.model.to(self.device)
                self.max_length = max_length
                self.max_seq_length = max_length
                self.query_instruction = (query_instruction or None)

            def encode(self, sentences, *, task_name: str | None = None, prompt_type=None, batch_size: int = 32, show_progress_bar: bool = True, **kwargs) -> np.ndarray:
                outs = []
                for i in range(0, len(sentences), batch_size):
                    batch = sentences[i:i+batch_size]
                    pn = kwargs.get("prompt_name", None)
                    is_query = (prompt_type == "query") or (pn == "query")
                    if is_query and self.query_instruction:
                        batch = [self.query_instruction + t for t in batch]
                    enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt", add_special_tokens=True)
                    input_ids = enc["input_ids"]
                    attention_mask = enc["attention_mask"]
                    lengths = attention_mask.sum(dim=1)
                    idx = lengths - 1
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    with torch.inference_mode():
                        out = self.model(**enc, return_dict=True)
                        hs = out.last_hidden_state
                        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
                        if left_padding:
                            pooled = hs[:, -1]
                        else:
                            pooled = hs[torch.arange(hs.size(0), device=hs.device), idx.to(hs.device)]
                        outs.append(pooled.detach().cpu())
                emb = torch.cat(outs, dim=0).float().numpy()
                return emb

            def encode_queries(self, queries, *, prompt_name: str | None = None, **kwargs):
                return self.encode(queries, prompt_type="query", prompt_name=prompt_name, **kwargs)

            def encode_corpus(self, corpus, *, prompt_name: str | None = None, **kwargs):
                if isinstance(corpus[0], dict):
                    texts = [f"{doc.get('title','')} {doc.get('text','')}".strip() for doc in corpus]
                else:
                    texts = corpus
                return self.encode(texts, prompt_type="text", prompt_name=prompt_name, **kwargs)

        model = Qwen3EOSWrapper(model_name, query_instruction=a.query_instruction)
        out_dir = compute_out_dir_qwen3eos(model_name, a.output_folder, a.query_instruction)
    else:
        print("not support model")
        return

    os.makedirs(out_dir, exist_ok=True)
    import time
    start = time.time()
    bs = a.batch_size
    encode_kwargs = {"batch_size": bs, "show_progress_bar": True}
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        model,
        output_folder=out_dir,
        encode_kwargs=encode_kwargs,
        limits=a.limits,
        overwrite_results=True,
    )
    dur = time.time() - start
    print(f"[MTEB] Done: model={a.model}, elapsed={dur:.2f}s, results_dir={out_dir}")


if __name__ == "__main__":
    main()
