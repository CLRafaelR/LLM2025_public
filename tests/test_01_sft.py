from unsloth import FastLanguageModel
import numpy as np
import torch
import os
import random
from typing import List
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from dotenv import load_dotenv
from scripts.document.write_readme import ReadmeWriter
from scripts.upload.upload_results import UploadResults
from scripts.monitor.callback import LabelStatsCallback
from datetime import datetime
import pytest
from scripts.preprocess.collate_data import AssistantOnlyCollatorCached, filter_has_supervision, count_all_masked
from scripts.preprocess.clean_data import (
    ensure_openai_messages,
    has_any_nonempty_assistant_turn,
    ends_with_nonempty_assistant,
    filter_by_token_length,
    apply_think_style,
    shuffle_split,
    make_text_cache_builder,
)
from scripts.preprocess.upsample_data import UpsampleConfig
from scripts.cleanup.release_memory import cleanup_training_run

# import wandb

load_dotenv()

hf_cache = os.getenv("HF_CACHE")

# wandb.login(key=os.getenv("WANDB_API_KEY"))

"""
固定パラメータ
"""


@pytest.fixture
def seed() -> int:
    return 20260225


@pytest.fixture(autouse=True)
def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def base_model_id() -> str:
    return str(os.getenv("SFT_BASE_MODEL"))


@pytest.fixture
def dataset_id() -> str:
    return str(os.getenv("SFT_DATASET_ID"))


@pytest.fixture
def val_ratio() -> float:
    return float(os.getenv("SFT_VAL_RATIO"))


@pytest.fixture
def per_device_train_batch_size() -> int:
    return int(os.getenv("SFT_PER_DEVICE_TRAIN_BATCH_SIZE"))


@pytest.fixture
def per_device_eval_batch_size() -> int:
    return int(os.getenv("SFT_PER_DEVICE_EVAL_BATCH_SIZE"))


@pytest.fixture
def grad_accum() -> int:
    return int(os.getenv("SFT_GRAD_ACCUM"))


@pytest.fixture
def warmup_ratio() -> float:
    return float(os.getenv("SFT_WARMUP_RATIO"))


@pytest.fixture
def weight_decay() -> float:
    return float(os.getenv("SFT_WEIGHT_DECAY"))


@pytest.fixture
def is_mask_cot() -> bool:
    return os.getenv("SFT_MASK_COT") in ("True", "true", "1")


@pytest.fixture
def output_markers() -> List[str]:
    return [s.strip() for s in os.getenv("SFT_OUTPUT_MARKERS").split(",") if s.strip()]


@pytest.fixture
def output_learn_mode() -> str:
    return str(os.getenv("SFT_OUTPUT_LEARN_MODE"))


@pytest.fixture
def logging_steps() -> int:
    return int(os.getenv("SFT_LOGGING_STEPS"))


@pytest.fixture
def eval_steps() -> int:
    return int(os.getenv("SFT_EVAL_STEPS"))


@pytest.fixture
def save_steps() -> int:
    return int(os.getenv("SFT_SAVE_STEPS"))


@pytest.fixture
def save_total_limit() -> int:
    return int(os.getenv("SFT_SAVE_TOTAL_LIMIT"))


@pytest.fixture
def max_steps() -> int:
    return int(os.getenv("SFT_MAX_STEPS"))


"""
変動パラメータ
"""


@pytest.mark.parametrize(
    argnames="think_style",
    argvalues=[
        "full",
        "tag_only",
        "remove",
    ],
)
@pytest.mark.parametrize(
    argnames="max_seq_len",
    argvalues=[
        512,
    ],
)
# LoRAランク
#
# 小さい値ほどパラメータ数が減りメモリ効率が上がるが、精度も低下する恐れがある
@pytest.mark.parametrize(
    argnames="lora_r",
    argvalues=[
        64,
    ],
)
@pytest.mark.parametrize(
    argnames="lora_alpha",
    argvalues=[
        64,
    ],
)
@pytest.mark.parametrize(
    argnames="lora_target_modules",
    argvalues=[
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",  # All
        "q_proj,k_proj,v_proj,o_proj",  # Attention only
        "q_proj,v_proj",  # Q and V only
        "gate_proj,up_proj,down_proj",  # MLP only
        "down_proj",  # The final MLP only
    ],
    ids=[
        "All",
        "Attention only",
        "Q and V only",
        "MLP only",
        "The final MLP only",
    ],
)
@pytest.mark.parametrize(
    argnames="lora_dropout",
    argvalues=[
        0,
    ],
)
@pytest.mark.parametrize(
    argnames="num_train_epochs",
    argvalues=[
        1,
    ],
)
@pytest.mark.parametrize(
    argnames="lr",
    argvalues=[
        1e-6,
    ],
)
def test_main(
    base_model_id,
    dataset_id,
    val_ratio,
    seed,
    think_style,
    max_seq_len,
    lora_r,
    lora_alpha,
    lora_target_modules,
    lora_dropout,
    num_train_epochs,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    grad_accum,
    lr,
    warmup_ratio,
    weight_decay,
    logging_steps,
    eval_steps,
    save_steps,
    save_total_limit,
    max_steps,
    is_mask_cot,
    output_markers,
    output_learn_mode,
):
    """テスト対象関数"""
    # try の直前で None 初期化（finally句でのUnboundLocalError 防止）
    model = tokenizer = trainer = collator = build_cache = None
    train_ds = val_ds = ds_all = args = None

    try:
        yymmdd_THHMMSS = str(datetime.now().strftime("%Y%m%d_T%H%M%S"))
        output_dir = os.getenv("OUTPUT_DIR_BASE") + "/" + yymmdd_THHMMSS
        os.makedirs(output_dir, exist_ok=True)

        adapter_id = str(base_model_id.split("/")[1] + "-" + yymmdd_THHMMSS)

        # Unslothでベースモデルを読み込む（4bitロードで省メモリ）
        cache_path_to_model = os.path.expanduser(
            hf_cache + "/models--unsloth--" + base_model_id.split("/")[1].lower() + "-unsloth-bnb-4bit"
        )
        if os.path.exists(cache_path_to_model):
            print(f"[INFO] Loading base model from local cache: {base_model_id}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_id,
                max_seq_length=max_seq_len,
                dtype=None,
                load_in_4bit=True,
                # cache_dir=cache_path_to_model,
                local_files_only=True,
            )
        else:
            print(f"[INFO] Downloading base model from Hugging Face Hub. It'll take few minutes...: {base_model_id}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_id,
                max_seq_length=max_seq_len,
                dtype=None,
                load_in_4bit=True,
            )

        cache_path_to_data = os.path.expanduser(
            hf_cache + "/datasets--" + dataset_id.split("/")[0].lower() + "--" + dataset_id.split("/")[1].lower()
        )
        if os.path.exists(cache_path_to_data):
            print(f"[INFO] Loading dataset from local cache: {dataset_id}")
            ds_all = load_dataset(
                dataset_id,
                split="train",
                cache_dir=cache_path_to_data,
            )
        else:
            print(f"[INFO] Loading dataset from Hugging Face Hub: {dataset_id}")
            ds_all = load_dataset(
                dataset_id,
                split="train",
            )

        # データ形式チェック（messagesがlistであること）
        ensure_openai_messages(ds_all)

        # 学習できるサンプルだけ残す（assistantが空なら教師信号が無い）
        ds_all = ds_all.filter(lambda ex: has_any_nonempty_assistant_turn(ex["messages"]))
        ds_all = ds_all.filter(ends_with_nonempty_assistant)
        ds_all = filter_by_token_length(ds_all, tokenizer)
        ds_all = apply_think_style(ds_all, think_style)

        # train/val分割
        train_ds, val_ds = shuffle_split(ds_all, val_ratio, seed)

        # Optional: upsampling by rule（分割後に適用）
        train_ds = UpsampleConfig().apply_upsampling(train_ds)

        # Cache chat template renders（tokenizerが必要なのでここで初めてbuild_cacheを作る）
        build_cache = make_text_cache_builder(tokenizer)

        train_ds = train_ds.map(
            build_cache,
            batched=True,
            num_proc=1,
            desc="Caching train",
        )
        val_ds = val_ds.map(
            build_cache,
            batched=True,
            num_proc=1,
            desc="Caching val",
        )

        # Attach LoRA
        # ここで「学習される部分（LoRAアダプタ）」をモデルに追加します。
        # 学習対象は LoRA のパラメータだけになり、ベースモデルの巨大な重みは固定されます。
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules.split(","),
            lora_dropout=lora_dropout,
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )

        # Transformersの引数名がバージョンで揺れることがあります。
        # 今回のバージョンでは eval_strategy を使います。
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            max_steps=max_steps,  # -1 => epoch-based
            bf16=False,
            fp16=True,  # T4向け（T4はbf16が弱いのでfp16を使うのが一般的）
            push_to_hub=False,
            group_by_length=False,
            remove_unused_columns=False,
            report_to="none",
            # report_to="wandb",
        )

        # assistant-only loss の collator を使う
        collator = AssistantOnlyCollatorCached(
            tokenizer=tokenizer,
            max_length=max_seq_len,
            is_mask_cot=is_mask_cot,
            output_markers=output_markers,
            output_learn_mode=output_learn_mode,
        )

        # --- NaN対策：all-masked（教師トークン0）を除去して評価を安定化 ---
        print("[INFO] Checking all-masked samples before filtering...")
        count_all_masked(val_ds, collator, n=len(val_ds), seed=seed)

        print("[INFO] Filtering train/val to remove all-masked samples...")
        train_ds = filter_has_supervision(train_ds, collator)
        val_ds = filter_has_supervision(val_ds, collator)

        print("[INFO] New sizes:", "train =", len(train_ds), "val =", len(val_ds))
        print("[INFO] Checking all-masked samples after filtering...")
        count_all_masked(val_ds, collator, n=len(val_ds), seed=seed)

        # Trainer（Transformersの標準学習ループ）
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            tokenizer=tokenizer,
        )

        # 監視用コールバックを追加（学習が効いているかのヘルスチェック）
        trainer.add_callback(
            LabelStatsCallback(
                train_ds,
                collator,
                name="train",
                every_n_steps=logging_steps,
            )
        )

        train_start = datetime.now()
        print(f"[INFO] Training started at {train_start.strftime('%Y-%m-%d %H:%M:%S')}")

        trainer.train()

        train_end = datetime.now()
        print(f"[INFO] Training ended at {train_end.strftime('%Y-%m-%d %H:%M:%S')}")

        elapsed = train_end - train_start
        total_seconds = int(elapsed.total_seconds())
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"[INFO] Training completed in {days}day, {hours:02d}:{minutes:02d}:{seconds:02d}")

        # 学習結果の保存：LoRAアダプタ＆tokenizer
        print("[INFO] Saving adapter & tokenizer...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[INFO] Done. Saved to {output_dir}")

        # 学習結果の保存：README.md
        ReadmeWriter(
            base_model_id=base_model_id,
            adapter_id=adapter_id,
            dataset_id=dataset_id,
            think_style=think_style,
            max_seq_len=max_seq_len,
            num_train_epochs=num_train_epochs,
            lr=lr,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            out_lora_dir=output_dir,
            lora_target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            per_device_train_batch_size=per_device_train_batch_size,
            grad_accum=grad_accum,
        ).run()

        UploadResults(
            output_dir=output_dir,
            stage_dir=os.getenv("STAGE_DIR"),
            hf_repo_id=str(os.getenv("HF_REPO_ID") + "/" + adapter_id),
            is_private=os.getenv("HF_PRIVATE") in ("True", "true", "1"),
        ).run()
    finally:
        # cleanup_training_run()で内部循環参照を切断する
        # - trainer内部循環参照切断
        # - unsloth属性除去
        # - optimizer等の解放
        cleanup_training_run(
            trainer=trainer,
            model=model,
            tokenizer=tokenizer,
            collator=collator,
            build_cache=build_cache,
            train_ds=train_ds,
            val_ds=val_ds,
            ds_all=ds_all,
            args=args,
        )

        # test_main のローカル変数がモデル等を参照し続けているため
        # モデル本体のGPUメモリは解放されない。ここで参照を落とす。
        del trainer, model, tokenizer, collator, build_cache
        del train_ds, val_ds, ds_all, args
