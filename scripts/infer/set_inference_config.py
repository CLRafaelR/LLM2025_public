def build_try_configs(model_source):

    # Primary presets

    if model_source == "merged":
        base = [
            {"max_model_len": 4096, "max_tokens": 4096, "gpu_mem": 0.85},
            {"max_model_len": 4096, "max_tokens": 4096, "gpu_mem": 0.80},
        ]
        # ↑ 4096トークンまでの文脈/出力を許しつつ、GPU使用率を0.85→0.80で試す

    elif model_source == "adapter_merge":
        base = [
            {"max_model_len": 4096, "max_tokens": 4096, "gpu_mem": 0.60},
            {"max_model_len": 4096, "max_tokens": 4096, "gpu_mem": 0.65},
        ]
        # ↑ adapter_merge はメモリが厳しくなりがちなので、gpu_memを低めから試します。

    else:  # base
        base = [
            {"max_model_len": 4096, "max_tokens": 4096, "gpu_mem": 0.80},
            {"max_model_len": 4096, "max_tokens": 4096, "gpu_mem": 0.70},
        ]
        # ↑ baseモデルは比較的軽い想定で、0.80→0.70を試します。

    # Fallback ladder (reduce context / output)
    # 失敗したときの「段階的に軽くする設定」。
    # max_model_len と max_tokens を下げると、必要メモリが減り成功しやすくなります。
    ladder = [
        {"max_model_len": 3072, "max_tokens": 3072},
        {"max_model_len": 2048, "max_tokens": 2048},
        {"max_model_len": 1536, "max_tokens": 1536},
    ]

    # Expand base configs with ladder and a couple gpu_mem tweaks
    # ↑ base設定に対し、ladder段階を「合成」して試行パターンを増やします。
    #   また、gpu_memも少し増やす版を試します（失敗理由が「確保不足」系のときに効く場合がある）。
    out = []
    for cfg in base:
        out.append(cfg)

        for step in ladder:
            out.append({**cfg, **step})

        # try a slightly higher gpu_mem if still failing (some failures are "not enough alloc")
        out.append({**cfg, "gpu_mem": min(0.90, cfg["gpu_mem"] + 0.05)})

    # Deduplicate while preserving order
    # ↑ 似た設定が重複し得るので、順序を保ったまま重複削除します。
    seen = set()
    uniq = []
    for c in out:
        key = (c["max_model_len"], c["max_tokens"], round(c["gpu_mem"], 2))

        if key in seen:
            continue

        seen.add(key)
        uniq.append(c)

    return uniq
