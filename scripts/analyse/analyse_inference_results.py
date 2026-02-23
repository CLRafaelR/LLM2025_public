"""
# Originally came from the following URL:
# https://file.notion.so/f/f/9db80869-7992-4b3a-8752-893d590f311d/f25fb33c-c383-4aeb-9ea4-075928491b22/inference_analyzer.py?table=block&id=3019a903-acd0-8011-ace9-cac53124cb35&spaceId=9db80869-7992-4b3a-8752-893d590f311d&expirationTimestamp=1771855200000&signature=0fjfbpVdoOD216LKzzHBqdK-tttFV1Vd9X6_1fl1mkE&downloadName=inference_analyzer.py

推論結果分析ツール
使い方：
1. このファイルと同じフォルダに inference.json と public_150.json を置く
2. python inference_analyzer.py を実行
"""

import json
import os


def load_files():
    """ファイルを読み込む"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    inference_path = os.path.join(script_dir, "inference.json")
    public_path = os.path.join(script_dir, "public_150.json")

    if not os.path.exists(inference_path):
        print("❌ inference.json が見つかりません")
        print(f"   このファイルと同じフォルダに置いてください: {script_dir}")
        return None, None

    if not os.path.exists(public_path):
        print("❌ public_150.json が見つかりません")
        print(f"   このファイルと同じフォルダに置いてください: {script_dir}")
        return None, None

    with open(inference_path, "r", encoding="utf-8") as f:
        inference = {item["task_id"]: item for item in json.load(f)}

    with open(public_path, "r", encoding="utf-8") as f:
        public = {item["task_id"]: item for item in json.load(f)}

    return inference, public


def analyze_toml(inference, public):
    """TOML の形式を詳細分析"""
    print("\n" + "=" * 60)
    print("📊 TOML 形式分析")
    print("=" * 60)

    standard = 0
    inline = 0
    other = 0

    for task_id, pub in public.items():
        if pub.get("output_type") == "TOML":
            gen = inference.get(task_id, {}).get("generation", "")

            if "[[" in gen:
                standard += 1
                label = "✅ 標準"
            elif "= {" in gen[:200]:
                inline += 1
                label = "⚠️ インライン"
            else:
                other += 1
                label = "❓ その他"

            print(f"\n{label}: {pub.get('task_name', 'Unknown')}")
            print("-" * 40)
            print(gen[:300] + "..." if len(gen) > 300 else gen)

    print("\n" + "-" * 40)
    print(f"【集計】")
    print(f"  標準形式 [[section]]: {standard}件")
    print(f"  インライン形式 {{}}: {inline}件")
    print(f"  その他: {other}件")


def analyze_all_formats(inference, public):
    """全フォーマットの概要"""
    print("\n" + "=" * 60)
    print("📊 全フォーマット概要")
    print("=" * 60)

    formats = {}

    for task_id, pub in public.items():
        fmt = pub.get("output_type", "Unknown")
        gen = inference.get(task_id, {}).get("generation", "")

        if fmt not in formats:
            formats[fmt] = {"total": 0, "valid": 0, "lengths": []}

        formats[fmt]["total"] += 1
        formats[fmt]["lengths"].append(len(gen))

        # 簡易有効性チェック
        valid = False
        if fmt == "JSON":
            try:
                json.loads(gen.strip())
                valid = True
            except:
                pass
        elif fmt == "YAML":
            valid = ":" in gen and not gen.strip().startswith("{")
        elif fmt == "XML":
            valid = gen.strip().startswith("<?xml") or gen.strip().startswith("<")
        elif fmt == "CSV":
            valid = "," in gen and "\n" in gen
        elif fmt == "TOML":
            valid = "=" in gen

        if valid:
            formats[fmt]["valid"] += 1

    for fmt, data in formats.items():
        avg_len = sum(data["lengths"]) / len(data["lengths"]) if data["lengths"] else 0
        print(f"\n【{fmt}】")
        print(f"  有効: {data['valid']}/{data['total']}")
        print(f"  平均文字数: {avg_len:.0f}")
        print(f"  最小: {min(data['lengths'])}, 最大: {max(data['lengths'])}")


def show_samples(inference, public, format_type="all", num_samples=2):
    """サンプル出力を表示"""
    print("\n" + "=" * 60)
    print(f"📄 サンプル出力（各{num_samples}件）")
    print("=" * 60)

    formats_to_show = ["JSON", "YAML", "XML", "CSV", "TOML"] if format_type == "all" else [format_type]

    for fmt in formats_to_show:
        print(f"\n{'=' * 20} {fmt} {'=' * 20}")
        count = 0
        for task_id, pub in public.items():
            if pub.get("output_type") == fmt:
                gen = inference.get(task_id, {}).get("generation", "")
                print(f"\n--- {pub.get('task_name', 'Unknown')} ({len(gen)}文字) ---")
                print(gen[:500] + "..." if len(gen) > 500 else gen)
                count += 1
                if count >= num_samples:
                    break


def compare_with_best(inference, public, best_inference):
    """BEST モデルとの比較"""
    print("\n" + "=" * 60)
    print("🔍 BEST モデルとの比較")
    print("=" * 60)

    for fmt in ["JSON", "YAML", "XML", "CSV", "TOML"]:
        current_lens = []
        best_lens = []

        for task_id, pub in public.items():
            if pub.get("output_type") == fmt:
                current_gen = inference.get(task_id, {}).get("generation", "")
                best_gen = best_inference.get(task_id, {}).get("generation", "")
                current_lens.append(len(current_gen))
                best_lens.append(len(best_gen))

        if current_lens and best_lens:
            current_avg = sum(current_lens) / len(current_lens)
            best_avg = sum(best_lens) / len(best_lens)
            diff = current_avg - best_avg
            emoji = "⬆️" if diff > 50 else "⬇️" if diff < -50 else "➡️"
            print(f"{fmt}: 現在 {current_avg:.0f}文字 vs BEST {best_avg:.0f}文字 {emoji} ({diff:+.0f})")


def main():
    print("🔬 推論結果分析ツール")
    print("=" * 60)

    inference, public = load_files()
    if inference is None:
        return

    print(f"✅ 読み込み完了: {len(inference)}件の推論結果")

    while True:
        print("\n" + "-" * 40)
        print("メニュー:")
        print("  1. 全フォーマット概要")
        print("  2. TOML 詳細分析")
        print("  3. サンプル出力を見る")
        print("  4. 特定フォーマットのサンプルを見る")
        print("  0. 終了")
        print("-" * 40)

        choice = input("選択 (0-4): ").strip()

        if choice == "0":
            print("👋 お疲れさまでした！")
            break
        elif choice == "1":
            analyze_all_formats(inference, public)
        elif choice == "2":
            analyze_toml(inference, public)
        elif choice == "3":
            show_samples(inference, public, "all", 2)
        elif choice == "4":
            fmt = input("フォーマット (JSON/YAML/XML/CSV/TOML): ").strip().upper()
            if fmt in ["JSON", "YAML", "XML", "CSV", "TOML"]:
                num = input("何件表示する？ (デフォルト2): ").strip()
                num = int(num) if num.isdigit() else 2
                show_samples(inference, public, fmt, num)
            else:
                print("❌ 無効なフォーマット")
        else:
            print("❌ 無効な選択")


if __name__ == "__main__":
    main()
