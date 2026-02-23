from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
from datasets import load_dataset
from scripts.preprocess.clean_data import (
    ensure_openai_messages,
    has_any_nonempty_assistant_turn,
    ends_with_nonempty_assistant,
    apply_think_style,
)

load_dotenv()

hf_cache = os.getenv("HF_CACHE")
dataset_id = os.getenv("SFT_DATASET_ID")
base_model_id = os.getenv("SFT_BASE_MODEL")


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
    ds_all = load_dataset(dataset_id, split="train")

cache_path_to_model = os.path.expanduser(
    hf_cache + "/models--unsloth--" + base_model_id.split("/")[1].lower() + "-unsloth-bnb-4bit"
)
if os.path.exists(cache_path_to_model):
    print(f"[INFO] Loading base model from local cache: {base_model_id}")
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        dtype=None,
        load_in_4bit=True,
        # cache_dir=cache_path_to_model,
        local_files_only=True,
    )
else:
    print(f"[INFO] Downloading base model from Hugging Face Hub. It'll take few minutes...: {base_model_id}")
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        dtype=None,
        load_in_4bit=True,
    )

# データ形式チェック（messagesがlistであること）
ensure_openai_messages(ds_all)

# 学習できるサンプルだけ残す（assistantが空なら教師信号が無い）
ds_all = ds_all.filter(lambda ex: has_any_nonempty_assistant_turn(ex["messages"]))
ds_all = ds_all.filter(ends_with_nonempty_assistant)


# --- トークン数カウント ---

# 会話レベル: apply_chat_template でトークン列を取得してカウント
conv_token_counts = np.array([len(tokenizer.apply_chat_template(row["messages"], tokenize=True)) for row in ds_all])

# メッセージレベル: role 別にトークン数をカウント
ROLES = ("system", "user", "assistant")
all_msgs = list(
    filter(
        lambda m: m.get("role") in set(ROLES) and m.get("content"),
        chain.from_iterable(row["messages"] for row in ds_all),
    )
)
role_token_counts = {
    role: np.fromiter(
        map(lambda m: len(tokenizer.encode(m["content"])), filter(lambda m: m["role"] == role, all_msgs)),
        dtype=int,
    )
    for role in ROLES
}


def print_stats(name: str, data: np.ndarray) -> None:
    print(f"\n--- {name} ---")
    print(f"  count : {len(data)}")
    print(f"  mean  : {np.mean(data):.2f}")
    print(f"  std   : {np.std(data):.2f}")
    print(f"  var   : {np.var(data):.2f}")
    print(f"  min   : {np.min(data)}")
    print(f"  max   : {np.max(data)}")


print_stats("Conversation-level token counts", conv_token_counts)
for role, counts in role_token_counts.items():
    print_stats(f"Message-level token counts (role={role})", counts)

# --- ヒストグラム描画 ---

os.makedirs("outputs", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

sns.histplot(conv_token_counts, kde=True, ax=axes[0])
axes[0].set_title("Conversation-level token counts")
axes[0].set_xlabel("Token count")
axes[0].set_ylabel("Frequency")

for ax, (role, counts) in zip(axes[1:], role_token_counts.items()):
    sns.histplot(counts, kde=True, ax=ax)
    ax.set_title(f"Message-level token counts (role={role})")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("outputs/token_count_histograms.png", dpi=150)
plt.show()
print("[INFO] Histogram saved to outputs/token_count_histograms.png")

# assistant トークン数が mean ± 2SD に収まる会話のみ残す
asst_mean = np.mean(role_token_counts["assistant"])
asst_std = np.std(role_token_counts["assistant"])
asst_low, asst_high = asst_mean - 2 * asst_std, asst_mean + 2 * asst_std

filtered_ds_all = ds_all.filter(
    lambda row: all(
        asst_low <= len(tokenizer.encode(msg["content"])) <= asst_high
        for msg in row["messages"]
        if msg.get("role") == "assistant" and msg.get("content")
    )
)
print(f"[INFO] Filtered dataset size: {len(ds_all)} -> {len(filtered_ds_all)}")

# --- filtered_ds_all のトークン数カウント ---

conv_token_counts_f = np.array(
    [len(tokenizer.apply_chat_template(row["messages"], tokenize=True)) for row in filtered_ds_all]
)

all_msgs_f = list(
    filter(
        lambda m: m.get("role") in set(ROLES) and m.get("content"),
        chain.from_iterable(row["messages"] for row in filtered_ds_all),
    )
)
role_token_counts_f = {
    role: np.fromiter(
        map(lambda m: len(tokenizer.encode(m["content"])), filter(lambda m: m["role"] == role, all_msgs_f)),
        dtype=int,
    )
    for role in ROLES
}

print("\n===== Filtered dataset stats =====")
print_stats("Conversation-level token counts (filtered)", conv_token_counts_f)
for role, counts in role_token_counts_f.items():
    print_stats(f"Message-level token counts (role={role}, filtered)", counts)

# --- filtered_ds_all ヒストグラム描画 ---

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

sns.histplot(conv_token_counts_f, kde=True, ax=axes[0])
axes[0].set_title("Conversation-level token counts (filtered)")
axes[0].set_xlabel("Token count")
axes[0].set_ylabel("Frequency")

for ax, (role, counts) in zip(axes[1:], role_token_counts_f.items()):
    sns.histplot(counts, kde=True, ax=ax)
    ax.set_title(f"Message-level token counts (role={role}, filtered)")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("outputs/token_count_histograms_filtered.png", dpi=150)
plt.show()
print("[INFO] Histogram saved to outputs/token_count_histograms_filtered.png")

# --- cleaned_ds_all: user≤400 かつ assistant≤500 トークンのみ ---

cleaned_ds_all = ds_all.filter(
    lambda row: (
        all(
            len(tokenizer.encode(msg["content"])) <= 400
            for msg in row["messages"]
            if msg.get("role") == "user" and msg.get("content")
        )
        and all(
            len(tokenizer.encode(msg["content"])) <= 500
            for msg in row["messages"]
            if msg.get("role") == "assistant" and msg.get("content")
        )
    )
)
print(f"[INFO] Cleaned dataset size: {len(ds_all)} -> {len(cleaned_ds_all)}")

conv_token_counts_c = np.array(
    [len(tokenizer.apply_chat_template(row["messages"], tokenize=True)) for row in cleaned_ds_all]
)

all_msgs_c = list(
    filter(
        lambda m: m.get("role") in set(ROLES) and m.get("content"),
        chain.from_iterable(row["messages"] for row in cleaned_ds_all),
    )
)
role_token_counts_c = {
    role: np.fromiter(
        map(lambda m: len(tokenizer.encode(m["content"])), filter(lambda m: m["role"] == role, all_msgs_c)),
        dtype=int,
    )
    for role in ROLES
}

print("\n===== Cleaned dataset stats =====")
print_stats("Conversation-level token counts (cleaned)", conv_token_counts_c)
for role, counts in role_token_counts_c.items():
    print_stats(f"Message-level token counts (role={role}, cleaned)", counts)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

sns.histplot(conv_token_counts_c, kde=True, ax=axes[0])
axes[0].set_title("Conversation-level token counts (cleaned)")
axes[0].set_xlabel("Token count")
axes[0].set_ylabel("Frequency")

for ax, (role, counts) in zip(axes[1:], role_token_counts_c.items()):
    sns.histplot(counts, kde=True, ax=ax)
    ax.set_title(f"Message-level token counts (role={role}, cleaned)")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("outputs/token_count_histograms_cleaned.png", dpi=150)
plt.show()
print("[INFO] Histogram saved to outputs/token_count_histograms_cleaned.png")

# --- role=user トークン数 < 50 のサンプル表示 ---

short_user_ds = ds_all.filter(
    lambda row: any(
        len(tokenizer.encode(msg["content"])) < 50
        for msg in row["messages"]
        if msg.get("role") == "user" and msg.get("content")
    )
)
print(f"\n[INFO] Rows with short user messages (< 50 tokens): {len(short_user_ds)}")

sampled = short_user_ds.shuffle(seed=42).select(range(min(25, len(short_user_ds))))


def format_msg(msg: dict) -> str:
    content = msg.get("content") or ""
    n_tokens = len(tokenizer.encode(content)) if content else 0
    return f"  [{msg.get('role', '?')}] ({n_tokens} tokens): {content}"


def format_sample(idx_row: tuple[int, dict]) -> str:
    i, row = idx_row
    return "\n".join([f"===== Sample {i + 1} =====", *map(format_msg, row["messages"])])


print("\n\n".join(map(format_sample, enumerate(sampled))))

# --- role=system トークン数が指定値のサンプル表示 ---


def sample_by_system_tokens(ds, n_tokens: int, n_samples: int = 2, seed: int = 42):
    subset = ds.filter(
        lambda row: any(
            len(tokenizer.encode(msg["content"])) == n_tokens
            for msg in row["messages"]
            if msg.get("role") == "system" and msg.get("content")
        )
    )
    sampled = subset.shuffle(seed=seed).select(range(min(n_samples, len(subset))))
    print(f"\n[INFO] system tokens == {n_tokens}: {len(subset)} rows, showing {len(sampled)}")
    print("\n\n".join(map(format_sample, enumerate(sampled))))


sample_by_system_tokens(ds_all, n_tokens=21)
sample_by_system_tokens(ds_all, n_tokens=23)

# --- think_style 変換サンプル表示 ---

for _style in ("full", "tag_only", "remove"):
    _ds = apply_think_style(ds_all, think_style=_style)
    _sampled = _ds.shuffle(seed=42).select(range(min(2, len(_ds))))
    print(f"\n{'=' * 60}")
    print(f"think_style={_style!r}  (showing {len(_sampled)} samples)")
    print("=" * 60)
    print("\n\n".join(map(format_sample, enumerate(_sampled))))
