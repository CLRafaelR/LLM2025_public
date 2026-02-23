#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from huggingface_hub import login
import os
import json
from pathlib import Path
from transformers import AutoTokenizer
from scripts.infer.find_model_to_use import resolve_model_path
from scripts.infer.set_inference_config import build_try_configs
from scripts.infer.run_model import run_with_config


def infer_and_generate_json(adapter_id, subdir) -> None:
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    login(hf_token)

    # ------------------------------------------------------------
    # 1) Config
    # ------------------------------------------------------------

    model_source = os.getenv("MODEL_SOURCE")  # "merged" | "base" | "adapter_merge"

    # base model (HF repo id or local path)
    # 学習時に使用したベースモデルを入れてください。
    base_model_id_or_path = os.getenv("SFT_BASE_MODEL")

    hf_repo_id = os.getenv("HF_REPO_ID")

    merged_model_id_or_path = None
    if model_source == "merged":
        # merged model (HF repo id or local path)
        # アダプタではなくマージモデルをアップロードした場合は、ここにIDをいれてください。
        # "merged"を選択した場合に記入
        merged_model_id_or_path = f"{hf_repo_id}/{adapter_id}"
    elif model_source == "adapter_merge":
        # adapter merge
        # あなたがHuggingFaceにアップロードしたアダプタのIDを入れてください。
        # "adapter_merge"を選択した場合に記入
        adapter_id = f"{hf_repo_id}/{adapter_id}"

    # merge済モデルの一時保存
    merged_local_dir = f"{os.getenv('MERGED_LOCAL_DIR_BASE')}/{adapter_id}"

    # 入力（150問）と出力（提出用）ファイルパスの指定
    input_path = os.getenv("INPUT_PATH")
    output_path = f"{os.getenv('OUTPUT_DIR_BASE')}/{subdir}/inference.json"

    temperature = float(os.getenv("TEMPERATURE"))

    # ### Step 3: vLLM 推論の実行と提出用JSONの生成
    # - `custom_inference.py` が生成され、それを実行します。
    # - 推論結果は `/content/StructEval/outputs/nonrenderable.json` に保存されます。
    # - `output` を `generation` に補完し、提出用ファイル `/content/inference.json` を出力します。
    # - 出力された`/content/inference.json` をダウンロードして、Omnicampusに提出してください。
    # ---

    # ------------------------------------------------------------
    # 2) Stable vLLM env (IMPORTANT: must be set BEFORE importing vllm)
    # ------------------------------------------------------------

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # vLLM内部でワーカープロセスを作る方式を "spawn" に固定します。
    # Colabなど一部環境では "fork" より安定しやすいことがあります。

    os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
    # vLLMのログレベル（INFO）を設定します。デバッグ時に有用です。

    # ------------------------------------------------------------
    # 3) Resolve model_path
    # ------------------------------------------------------------

    # 最終的に使うモデルのパス/IDを確定
    model_path = resolve_model_path(
        model_source,
        base_model_id_or_path,
        merged_model_id_or_path,
        adapter_id,
        merged_local_dir,
    )
    print("[INFO] Using model:", model_path)

    # ------------------------------------------------------------
    # 4) Load public_150 and build prompts (no torch usage here)
    # ------------------------------------------------------------
    # 入力ファイルを読み込み、各問題の「プロンプト（モデルに渡す文字列）」を作ります。

    pub = json.loads(Path(input_path).read_text(encoding="utf-8"))

    assert isinstance(pub, list), "public_150.json must be a list"
    assert len(pub) == 150, f"public_150 must have 150 items, got {len(pub)}"
    assert len({x["task_id"] for x in pub}) == 150, "public_150 has duplicate task_id"

    # Safety: ensure output_type exists (office enriched file)

    missing_ot = [x.get("task_id") for x in pub if not (x.get("output_type") or "").strip()]

    if missing_ot:
        raise RuntimeError(f"FATAL: public_150 missing output_type (not enriched). Examples: {missing_ot[:5]}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # task_ids: 出力に使う task_id の並びを保存
    # prompts:   vLLMに渡すプロンプト文字列を保存
    task_ids, prompts = [], []

    for item in pub:
        task_ids.append(item["task_id"])
        query = item.get("query", "")
        messages = [{"role": "user", "content": query}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        # ↑ apply_chat_template で「モデルが期待する会話形式の文字列」に整形
        #   tokenize=False : まだトークン化せず、文字列として返す
        #   add_generation_prompt=True : 「ここからアシスタントが答える」境界を追加
        #   これにより、モデルが回答を続けて生成しやすい形になります。

    # ------------------------------------------------------------
    # 5) Presets + fallback plan
    # ------------------------------------------------------------
    # vLLM起動時に「文脈長(max_model_len)」や「出力上限(max_tokens)」を大きくしすぎると、
    # GPUメモリ不足(OOM)で落ちやすいです。
    # そこで、成功しやすい設定をいくつか用意し、失敗したら段階的に軽くして再試行します。
    # merged（既に焼き込み済み）と adapter_merge（その場でマージ）では、
    # 実メモリ使用量が変わることがあるため、最初に試す設定（gpu_memなど）を変えています。
    # 事前に「試行候補リスト」を作り、上から順に試します。

    try_configs = build_try_configs(model_source=model_source)
    # ↑ 実際に試す設定リストを作成します。

    print("[INFO] Try configs (in order):")

    for i, c in enumerate(try_configs[:8], 1):
        print(f"  {i:02d}. max_model_len={c['max_model_len']} max_tokens={c['max_tokens']} gpu_mem={c['gpu_mem']}")

    if len(try_configs) > 8:
        print(f"  ... total {len(try_configs)} configs")

    # ------------------------------------------------------------
    # 6) vLLM run with retry
    # ------------------------------------------------------------
    # ↑ ここからが推論本体です。

    last_err = None
    submission = None
    # ↑ 成功した場合に提出データ（150件）を入れる変数。成功まではNone。

    for idx, cfg in enumerate(try_configs, 1):
        print(
            f"[INFO] Attempt {idx}/{len(try_configs)}: max_model_len={cfg['max_model_len']} max_tokens={cfg['max_tokens']} gpu_mem={cfg['gpu_mem']}"
        )
        try:
            submission = run_with_config(
                cfg,
                temperature,
                model_path,
                prompts,
                task_ids,
            )
            print("[INFO] ✅ Generation succeeded with this config.")
            # ↑ 成功ログ
            break
        except RuntimeError as e:
            last_err = e
            msg = str(e)
            print("[WARN] Failed:", msg[:200].replace("\n", " "))

    # try next config
    if submission is None:
        raise RuntimeError(f"All configs failed. Last error: {last_err}")

    # Final guards
    # ↑ 最後に「提出物としての整合性チェック」をします。

    if len(submission) != 150:
        # ↑ 150件生成できているかチェック
        raise RuntimeError(f"Submission count mismatch: {len(submission)}")

    if len({x["task_id"] for x in submission}) != 150:
        # ↑ task_id の重複がないかチェック
        raise RuntimeError("Duplicate task_id in submission")

    Path(output_path).write_text(
        json.dumps(
            submission,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    # ↑ submission（Pythonオブジェクト）をJSON文字列にしてファイルへ保存します。

    print("[OK] wrote:", output_path, "items=150")
