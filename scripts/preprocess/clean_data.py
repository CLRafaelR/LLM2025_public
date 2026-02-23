import re
from typing import Any, Dict, List, Tuple
from datasets import Dataset


def ensure_openai_messages(ds: Dataset, msg_col: str = "messages") -> None:
    # データが「messages: [{role, content}, ...]」形式かをチェックします。
    # これは ChatGPT形式（OpenAIのChat Completions形式に似た）で、
    # tokenizer.apply_chat_template で安全に文字列化するために必要です。
    row0 = ds[0]
    ex = row0.get(msg_col, None)
    if not isinstance(ex, list):
        raise ValueError(f"Dataset must have list-style 'messages'. Got {type(ex)}")


def has_any_nonempty_assistant_turn(msgs: List[Dict[str, Any]]) -> bool:
    # “assistantの発話が空じゃない”ものが1回でも含まれるか？
    # SFTでは「正解例（assistantの出力）」がないと学習できないため。
    return any(m.get("role") == "assistant" and str(m.get("content", "")).strip() != "" for m in msgs)


def ends_with_nonempty_assistant(ex: Dict[str, Any]) -> bool:
    # 最後のターンが assistant の回答になっているサンプルだけを使います。
    # こうしておくと「最後のassistantだけ学習する（assistant-only loss）」設計と相性が良いです。
    msgs = ex.get("messages", [])
    if not msgs or msgs[-1].get("role") != "assistant":
        return False
    c = msgs[-1].get("content", "")
    return isinstance(c, str) and c.strip() != ""


def shuffle_split(ds: Dataset, val_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    # データをシャッフルして train/val に分割します。
    # val（検証）を持つことで「学習が進むほど性能が上がっているか／過学習していないか」を見られます。
    ds_shuf = ds.shuffle(seed=seed)
    n = len(ds_shuf)
    n_val = max(1, int(round(n * val_ratio)))
    return ds_shuf.select(range(n_val, n)), ds_shuf.select(range(n_val))


def make_text_cache_builder(tokenizer):
    # messages形式 → 実際にモデルに入力する“1本のテキスト”へ変換する関数を作ります。さらに「トークン長（truncationなし）」もキャッシュします。
    #
    # full_text  : ユーザー＋アシスタント（正解）まで含んだ全文
    # prefix_text: “最後のassistantの直前まで”の文（＝ここからassistantを生成させたい）
    #
    # この2つを持つことで、後のcollatorで「assistant部分だけをloss対象にする境界」を計算できます。

    def _build(batch):
        full_out = []
        prefix_out = []
        full_len_out = []
        prefix_len_out = []

        for msgs in batch["messages"]:
            full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            prefix = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)

            full_out.append(full)
            prefix_out.append(prefix)

            # 重要：ここで truncation=False で token 長だけ計算してキャッシュする
            # add_special_tokens=False はあなたの現行設計に合わせる（テンプレ側で必要トークンが入る想定）
            full_ids = tokenizer(full, add_special_tokens=False, truncation=False)["input_ids"]
            prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"]

            full_len_out.append(len(full_ids))
            prefix_len_out.append(len(prefix_ids))

        return {
            "full_text": full_out,
            "prefix_text": prefix_out,
            "full_input_ids_len": full_len_out,
            "prefix_input_ids_len": prefix_len_out,
        }

    return _build


def filter_by_token_length(
    ds: Dataset,
    tokenizer,
    max_user_tokens: int = 400,
    max_assistant_tokens: int = 500,
) -> Dataset:
    # role=user のトークン数が max_user_tokens 以下、かつ
    # role=assistant のトークン数が max_assistant_tokens 以下の行だけを残す。
    # どちらの条件も「その会話内の全メッセージ」に対して満たす必要がある。
    return ds.filter(
        lambda row: (
            all(
                len(tokenizer.encode(msg["content"])) <= max_user_tokens
                for msg in row["messages"]
                if msg.get("role") == "user" and msg.get("content")
            )
            and all(
                len(tokenizer.encode(msg["content"])) <= max_assistant_tokens
                for msg in row["messages"]
                if msg.get("role") == "assistant" and msg.get("content")
            )
        )
    )


def apply_think_style(ds: Dataset, think_style: str) -> Dataset:
    """role=assistant の content の Approach:/Output: ブロックを変換する。

    think_style:
        "full"     : Approach: -> <think>、\nOutput: -> </think>\n
        "tag_only" : Approach:...Output:\n を <think>\n</think>\n に置換
        "remove"   : Approach:...Output:\n を "" に置換（思考過程を完全削除）
    """
    if think_style not in ("full", "tag_only", "remove"):
        raise ValueError(f"Unknown think_style: {think_style!r}")

    def transform_content(content: str) -> str:
        if think_style == "full":
            return content.replace("Approach:", "<think>").replace("\nOutput:", "</think>\n")
        return re.sub(
            r"Approach:.*?\nOutput:\n",
            "<think>\n</think>\n" if think_style == "tag_only" else "",
            content,
            flags=re.DOTALL,
        )

    def transform_row(row: dict) -> dict:
        return {
            "messages": [
                {**msg, "content": transform_content(msg["content"])}
                if msg.get("role") == "assistant" and msg.get("content")
                else msg
                for msg in row["messages"]
            ]
        }

    return ds.map(transform_row)
