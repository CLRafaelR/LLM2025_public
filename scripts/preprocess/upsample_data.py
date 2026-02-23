from dataclasses import dataclass, field
from datasets import Dataset
import numpy as np
from typing import Dict


# -----------------------------
# 2.4) Optional upsampling
# -----------------------------
# upsamplingは「特定の種類のデータを多めに学習させる」テクニックです。
# 例：
# - JSONは得意だがYAMLは苦手 → YAML関連サンプルを2倍にする
# - 特定のsubcategoryが点数に効く → そこを厚くする
# ただし、やりすぎると他が弱くなることもあります（トレードオフ）。
# 学習データセットの品質が悪い等の原因で、却って性能が低下することもあります。
# その場合、学習データセットを観察し、追加の前処理が有効であることも多いです。


@dataclass
class UpsampleConfig:
    upsample_enable: bool = False
    upsample_rules_json: Dict[str, float] = field(default_factory=dict)

    def apply_upsampling(self, train_ds: Dataset) -> Dataset:
        if not self.upsample_enable or not self.upsample_rules_json:
            return train_ds
        rules = self.upsample_rules_json

        packs = train_ds["subcategory"] if "subcategory" in train_ds.column_names else [None] * len(train_ds)
        pack_field = train_ds["pack"] if "pack" in train_ds.column_names else [None] * len(train_ds)

        w = []
        for sub, pk in zip(packs, pack_field):
            weight = 1.0
            ssub = str(sub or "")
            spk = str(pk or "")
            for pat, mult in rules.items():
                try:
                    m = float(mult)
                except Exception:
                    m = 1.0
                if pat.startswith("pack:"):
                    if spk == pat.split(":", 1)[1]:
                        weight *= max(0.0, m)
                else:
                    if pat in ssub:
                        weight *= max(0.0, m)
            w.append(weight)

        w = np.asarray(w, dtype=np.float64)
        if (w <= 0).all() or w.sum() == 0:
            return train_ds

        p = w / w.sum()
        n = len(train_ds)
        idx = np.random.choice(np.arange(n), size=n, replace=True, p=p)
        print("[UPSAMPLE] rules:", rules)
        return train_ds.select(idx.tolist())
