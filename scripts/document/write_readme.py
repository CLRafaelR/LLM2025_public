# -----------------------------
# README.md（モデルカード）を out_lora_dir に生成
# -----------------------------
# 学習完了後に実行し、Hugging Face の README.md（モデルカード）を生成
# ベースモデル名・データセット名・学習ハイパーパラメータはコードの変数から自動同期

import os


# ------------------------------------------------------------------
# 補助関数の定義
# ------------------------------------------------------------------
def _s(x, default=""):
    try:
        v = str(x)
        return v if v.strip() else default
    except Exception:
        return default


def _fmt_lr(x) -> str:
    """
    Learning Rate の表記を整えるための関数。

    - 数値として解釈できる場合：
      指数表記（例: 1e-6）に整形する
    - 数値として解釈できない場合：
      元の値をそのまま文字列として出力する
      （誤った値を生成しないための安全策）
    """
    try:
        return f"{float(x):.0e}"
    except Exception:
        return _s(x, "")


# ------------------------------------------------------------------
# ReadmeWriter クラス
# ------------------------------------------------------------------
class ReadmeWriter:
    # README 内に記載するモデルタイトル
    # 変更したい場合は README.md を手書きで調整

    def __init__(
        self,
        base_model_id,
        adapter_id,
        dataset_id,
        think_style,
        max_seq_len,
        num_train_epochs,
        lr,
        lora_r,
        lora_alpha,
        out_lora_dir,
        lora_target_modules,
        lora_dropout,
        per_device_train_batch_size,
        grad_accum,
    ):
        # ------------------------------------------------------------------
        # 学習コードの変数から値を取得（README と自動同期）
        # ------------------------------------------------------------------
        self.base_model_id = _s(base_model_id, "Qwen/Qwen3-4B-Instruct-2507")
        self.adapter_id = adapter_id
        self.dataset_id = _s(
            dataset_id, "https://huggingface.co/datasets/u-10bei/structured_data_with_cot_dataset_512_v2"
        )

        self.think_style = str(think_style)

        self.max_seq_len = int(max_seq_len)
        self.epochs = int(num_train_epochs)
        self.lr_str = _fmt_lr(lr)

        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)

        self.out_lora_dir = out_lora_dir

        self.lora_target_modules = lora_target_modules
        self.lora_dropout = lora_dropout
        self.per_device_train_batch_size = per_device_train_batch_size
        self.grad_accum = grad_accum

        # NOTE:
        # - YAML front matter の license は
        #   「この LoRA アダプタ（リポジトリ）のライセンス表明」を意味する。
        # - 必要に応じて環境変数で差し替え可能。
        self.repo_license = os.environ.get("SFT_REPO_LICENSE", "apache-2.0")

    @property
    def _think_style_description(self) -> str:
        """Return a human-readable description of the think_style setting.

        Returns:
            A string describing how reasoning CoT and think tags were handled
            in the training data.
        """
        _descriptions = {
            "full": (
                "Reasoning CoT text was stored inside `<think>...</think>` tags "
                "and included in the training data as-is."
            ),
            "tag_only": (
                "Reasoning CoT text was removed, and only empty `<think>\\n</think>` "
                "tags were retained in the training data."
            ),
            "remove": (
                "Reasoning CoT text was removed entirely, and no `<think>` tags were added to the training data."
            ),
        }
        return _descriptions.get(self.think_style, self.think_style)

    def run(self):
        # ------------------------------------------------------------------
        # README.md 本文の生成
        # （説明テキストに準拠し、変数部分のみを自動置換）
        # ------------------------------------------------------------------
        readme_md = f"""---
base_model: {self.base_model_id}
datasets:
- {self.dataset_id}
language:
- en
license: {self.repo_license}
library_name: peft
pipeline_tag: text-generation
tags:
- qlora
- lora
- structured-output
---

# {self.adapter_id}

This repository provides a **LoRA adapter** fine-tuned from
**{self.base_model_id}** using **QLoRA (4-bit, Unsloth)**.

This repository contains **LoRA adapter weights only**.
The base model must be loaded separately.

## Training Objective

This adapter is trained to improve **structured output accuracy**
(JSON / YAML / XML / TOML / CSV).

Loss is applied only to the final assistant output,
while intermediate reasoning (Chain-of-Thought) is masked.

## Training Configuration

- Base model: {self.base_model_id}
- Method: QLoRA (4-bit)
- Max sequence length: {self.max_seq_len}
- Epochs: {self.epochs}
- Learning rate: {self.lr_str}
- LoRA: r={self.lora_r}, alpha={self.lora_alpha}
- Lora target modules: {self.lora_target_modules}
- Lora dropout: {self.lora_dropout}
- Per device batch size: {self.per_device_train_batch_size}
- Gradient accumulation step: {self.grad_accum}

## Training data

- Used dataset: {self.dataset_id}
- Treatment of reasoning CoT: {self._think_style_description}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "{self.base_model_id}"
adapter = "your_id/your-repo"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter)
```

## Sources & Terms (IMPORTANT)

Training data: {self.dataset_id}

Dataset License: MIT License. This dataset is used and distributed under the terms of the MIT License.
Compliance: Users must comply with the MIT license (including copyright notice) and the base model's original terms of use.
"""
        # ------------------------------------------------------------------
        # README.md の書き込み
        # ------------------------------------------------------------------

        readme_path = os.path.join(self.out_lora_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_md)

        # ------------------------------------------------------------------
        # 動作確認
        # ------------------------------------------------------------------

        assert os.path.exists(readme_path), "README.md was not written."
        assert readme_md.lstrip().startswith("---\n"), "README.md must start with YAML front matter."
        # 修正: 先頭の --- は改行なしで始まるため count("\n---\n") には含まれない。
        # そのため、閉じタグの分として 1回以上あればOKとする。
        assert readme_md.count("\n---\n") >= 1, "YAML front matter must be closed properly."

        print(f"[INFO] README.md written to: {readme_path}")
        print("[INFO] Preview (first 30 lines):")
        for i, line in enumerate(readme_md.splitlines()[:30], start=1):
            print(f"{i:02d}: {line}")
