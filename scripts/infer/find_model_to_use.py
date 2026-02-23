import os, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def resolve_model_path(
    model_source,
    base_model_id_or_path,
    merged_model_id_or_path,
    adapter_id,
    merged_local_dir,
):
    """どのモデルを使うかに応じて、vLLMへ渡すパス/IDを返す関数
    選んだmodel_sourceに応じて、最終的にvLLMに渡す「モデルの場所(model_path)」を決める
    """

    if model_source == "base":
        return base_model_id_or_path

    if model_source == "merged":
        return merged_model_id_or_path

    if model_source == "adapter_merge":
        # NOTE: torch/CUDA（GPU）を触るため、vLLMを起動する前に済ませます。
        print("[INFO] Merging adapter into base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id_or_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        # ベースモデルに対応するトークナイザを読み込み（マージ後も同じものを使うのが通常）
        tokenizer = AutoTokenizer.from_pretrained(base_model_id_or_path, trust_remote_code=True)

        # base_model に LoRAアダプタ(ADAPTER_ID) をマージ
        # merge後はLoRA層を外せるので（unload）、推論時の扱いが単純になります。
        model_to_merge = PeftModel.from_pretrained(base_model, adapter_id)
        merged_model = model_to_merge.merge_and_unload()

        os.makedirs(merged_local_dir, exist_ok=True)
        merged_model.save_pretrained(merged_local_dir)
        tokenizer.save_pretrained(merged_local_dir)

        del base_model, model_to_merge, merged_model
        gc.collect()
        torch.cuda.empty_cache()
        print("[INFO] Merged model saved:", merged_local_dir)
        return merged_local_dir

    raise ValueError("model_source must be 'merged'|'base'|'adapter_merge'")
