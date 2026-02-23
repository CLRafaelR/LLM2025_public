from dataclasses import dataclass
from typing import Any, Dict, List
import random
import torch

# -----------------------------
# 2.3) Collator (assistant-only loss)
# -----------------------------
# collatorは「生のサンプル群 → 学習に必要なテンソル(input_ids/labels等)」に変換する部品です。
#
# ここがこの学習コードの“設計思想”の核心：
# - 入力（user/system）も含めてモデルには読ませる
# - ただし loss（誤差）を計算するのは assistant の出力部分だけ
#
# これにより：
# - 「プロンプトを丸暗記させる」方向に学習が引っ張られにくい
# - “回答の形式”や“出力の正確さ”に学習の力点を置きます。

# 使用データセットによる仕様の違い
# データセット1：Output: が 100% なので CoT マスクが常に動き、Output本体だけ学習
# データセット2：Output: 系ラベルが存在しないため、CoTマスクは発動せず、“出力本体”を学習

# --- CoT mask settings (env overridable) ---

# MASK_COT = _getenv("SFT_MASK_COT", "1") in ("1","true","True")
# OUTPUT_MARKERS = [s.strip() for s in _getenv(
# "SFT_OUTPUT_MARKERS",
# "Output:,OUTPUT:,Final:,Answer:,Result:,Response:"
# ).split(",") if s.strip()]
# OUTPUT_LEARN_MODE = _getenv("SFT_OUTPUT_LEARN_MODE", "after_marker")  # after_marker / from_marker


@dataclass
class AssistantOnlyCollatorCached:
    tokenizer: Any
    max_length: int  # MAX_SEQ_LEN
    is_mask_cot: bool
    output_markers: List[str]
    output_learn_mode: str

    def _find_subsequence(self, seq: List[int], sub: List[int]) -> int:
        if not sub or len(sub) > len(seq):
            return -1
        for i in range(0, len(seq) - len(sub) + 1):
            if seq[i : i + len(sub)] == sub:
                return i
        return -1

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        tok = self.tokenizer
        full_texts = [ex["full_text"] for ex in batch]
        prefix_texts = [ex["prefix_text"] for ex in batch]

        old_trunc = getattr(tok, "truncation_side", "right")
        old_pad = getattr(tok, "padding_side", "right")
        tok.truncation_side = "left"
        tok.padding_side = "right"

        try:
            full_enc_tr = tok(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            input_ids = full_enc_tr["input_ids"]
            attention_mask = full_enc_tr["attention_mask"]
            labels = torch.full_like(input_ids, fill_value=-100)

            full_ids_nt = tok(
                full_texts, return_tensors=None, padding=False, truncation=False, add_special_tokens=False
            )["input_ids"]
            prefix_ids_nt = tok(
                prefix_texts, return_tensors=None, padding=False, truncation=False, add_special_tokens=False
            )["input_ids"]

            marker_token_seqs = []
            if self.is_mask_cot and self.output_markers:
                for m in self.output_markers:
                    mid = tok(m, add_special_tokens=False, truncation=False)["input_ids"]
                    if not mid:
                        continue
                    mid_nl = tok(m + "\n", add_special_tokens=False, truncation=False)["input_ids"]
                    mid_crlf = tok(m + "\r\n", add_special_tokens=False, truncation=False)["input_ids"]
                    marker_token_seqs.append((mid, mid_nl, mid_crlf))

            for i in range(input_ids.size(0)):
                trunc_left = max(0, len(full_ids_nt[i]) - self.max_length)
                boundary = len(prefix_ids_nt[i]) - trunc_left
                full_len_tr = int(attention_mask[i].sum().item())

                # assistant開始が見えていない => 学習対象外（元コード方針を維持）
                if boundary <= 0 or boundary >= full_len_tr:
                    continue

                span_start = boundary
                span_end = full_len_tr

                # デフォルト：assistant全体を学習（データセット2はここに落ちる）
                learn_start = span_start

                # CoTマスク：Output marker が見つかったときだけ学習開始点を進める（データセット1で発動）
                if self.is_mask_cot and marker_token_seqs:
                    visible_ids = input_ids[i, :full_len_tr].tolist()
                    assistant_ids = visible_ids[span_start:span_end]

                    best_out = None  # (out_pos, after_pos)
                    for mid, mid_nl, mid_crlf in marker_token_seqs:
                        # 改行付き優先
                        p = self._find_subsequence(assistant_ids, mid_nl)
                        if p != -1:
                            out_pos = span_start + p
                            after_pos = out_pos + len(mid_nl)
                        else:
                            p = self._find_subsequence(assistant_ids, mid_crlf)
                            if p != -1:
                                out_pos = span_start + p
                                after_pos = out_pos + len(mid_crlf)
                            else:
                                p = self._find_subsequence(assistant_ids, mid)
                                if p == -1:
                                    continue
                                out_pos = span_start + p
                                after_pos = out_pos + len(mid)

                        if (best_out is None) or (out_pos < best_out[0]):
                            best_out = (out_pos, after_pos)

                    if best_out is not None:
                        out_pos, after_pos = best_out
                        if self.output_learn_mode == "from_marker":
                            learn_start = out_pos
                        else:
                            learn_start = after_pos
                        learn_start = max(span_start, min(learn_start, span_end))

                if learn_start < span_end:
                    labels[i, learn_start:span_end] = input_ids[i, learn_start:span_end]

            labels[attention_mask == 0] = -100
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        finally:
            tok.truncation_side = old_trunc
            tok.padding_side = old_pad


@torch.no_grad()
def filter_has_supervision(ds, collator):
    keep = []
    for i in range(len(ds)):
        out = collator([ds[i]])
        if (out["labels"][0] != -100).sum().item() > 0:
            keep.append(i)
    return ds.select(keep)


def count_all_masked(ds, collator, n=200, seed=3407):
    rng = random.Random(seed)
    n = min(n, len(ds))
    idxs = [rng.randrange(0, len(ds)) for _ in range(n)]
    all_masked = 0
    for i in idxs:
        out = collator([ds[i]])
        labels = out["labels"][0]
        if (labels != -100).sum().item() == 0:
            all_masked += 1
    print(f"[CHECK] all-masked samples in {n}: {all_masked} ({all_masked / max(1, n):.1%})")
