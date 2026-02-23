from transformers import TrainerCallback
import torch
import random


# -----------------------------
# 2.5) Callback (monitor)
# -----------------------------
# 学習中のデバッグ用コールバックです。
# ここでは「labelsのうち、実際にloss対象になっているトークン割合」を時々表示します。
#
# 意味：
# - valid_ratio が極端に小さい → “学習していない”のと同じ（ラベルがほぼ -100）
# - valid_ratio が適度にある → assistant部分にしっかりlossが乗っている
#
# 初学者向けに言うと：
# - これは“学習がちゃんと効いているかの健康診断”です。


class LabelStatsCallback(TrainerCallback):
    def __init__(self, dataset, collator, name="train", every_n_steps=100):
        self.dataset, self.collator, self.name, self.every_n_steps = dataset, collator, name, every_n_steps

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step % self.every_n_steps) == 0:
            batch = [self.dataset[random.randint(0, len(self.dataset) - 1)] for _ in range(8)]
            out = self.collator(batch)
            valid = (out["labels"] != -100).sum().item()
            total = (out["attention_mask"] == 1).sum().item()
            print(f"\n[LabelStats:{self.name}] step={state.global_step} valid_ratio={valid / max(1, total):.4f}")
