import gc

import torch
from vllm import LLM, SamplingParams


def run_with_config(cfg, temperature, model_path, prompts, task_ids):

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=cfg["max_tokens"],
    )

    llm = LLM(
        model=model_path,
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=cfg["gpu_mem"],
        enforce_eager=True,
        tensor_parallel_size=1,
        disable_log_stats=True,
    )

    outs = llm.generate(prompts, sampling)

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    submission = []
    # ↑ 提出形式 [{"task_id": ..., "generation": ...}, ...] を作ります。

    for tid, out in zip(task_ids, outs):
        gen = out.outputs[0].text if out.outputs else ""
        submission.append({"task_id": tid, "generation": gen})
    return submission
    # ↑ 150問ぶんの提出配列を返します。
