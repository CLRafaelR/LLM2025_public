# ============================================================
# 3) LoRAアダプターをHugging Faceへアップロード (作成済みのREADMEを含む)
# ============================================================

import os
import fnmatch
import shutil
from pathlib import Path
from huggingface_hub import login, HfApi
from dotenv import load_dotenv


# ------------------------------------------------------------------
# 補助関数の定義
# ------------------------------------------------------------------
ALLOW_PATTERNS = [
    "README.md",
    "adapter_config.json",
    "adapter_model.*",
    "tokenizer.*",
    "special_tokens_map.json",
    "*.json",
]


def is_allowed(name: str) -> bool:
    """ファイル名が許可パターンに一致するか判定する関数"""
    return any(fnmatch.fnmatch(name, pat) for pat in ALLOW_PATTERNS)


# ------------------------------------------------------------------
# UploadResults クラス
# ------------------------------------------------------------------
class UploadResults:
    def __init__(self, output_dir, stage_dir, hf_repo_id, is_private):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        login(hf_token)
        self.api = HfApi()

        self.lora_save_dir = Path(output_dir)  # 学習済みモデルが保存されているディレクトリ
        self.stage_dir = Path(stage_dir)  # アップロード用の一時フォルダ（ステージング領域）
        self.hf_repo_id = hf_repo_id  # アップロード先のレポジトリID
        self.is_private = is_private  # 非公開設定

    def run(self):
        # -----------------------------
        # 3.1) 必須ファイルの存在確認
        # -----------------------------
        # アップロードに最低限必要なファイルを定義します
        required_files = {
            "adapter_config.json",  # LoRAの設定ファイル
            "README.md",  # 受講生が作成した解説文書
        }

        # 保存ディレクトリにあるファイル名のリストを取得
        present = {p.name for p in self.lora_save_dir.iterdir() if p.is_file()}

        # 足りないファイルをリストアップ
        missing = [f for f in required_files if f not in present]

        # モデル本体（adapter_model.safetensors または .bin）が存在するか確認
        if not any(f.startswith("adapter_model.") for f in present):
            missing.append("adapter_model.(safetensors|bin)")

        # 必須ファイルが欠けている場合は、エラーを表示して処理を中断します
        if missing:
            raise RuntimeError(
                "アップロードを中止しました。\n"
                "以下の必須ファイルが見つかりません:\n"
                + "\n".join(f"- {m}" for m in missing)
                + "\n\nアップロード前に、README.md を手書きで作成し保存してください。"
            )

        print("✅ 必須ファイルを確認できました")

        # -----------------------------
        # 3.2) アップロード対象の選別（ホワイトリスト）
        # -----------------------------
        # 不要な一時ファイルなどをアップロードしないよう、許可するファイル形式を指定します

        if self.stage_dir.exists():
            shutil.rmtree(self.stage_dir)  # 既存のフォルダがあれば一旦削除
        self.stage_dir.mkdir(parents=True)

        # 許可されたファイルだけを一時フォルダにコピー
        for p in self.lora_save_dir.iterdir():
            if p.is_file() and is_allowed(p.name):
                (self.stage_dir / p.name).write_bytes(p.read_bytes())

        print("📦 アップロード対象ファイル:", [p.name for p in self.stage_dir.iterdir()])

        # -----------------------------
        # 3.3) リポジトリ作成とアップロード
        # -----------------------------

        # Hugging Face上にリポジトリを作成（既に存在していてもOK）
        self.api.create_repo(
            repo_id=self.hf_repo_id,
            repo_type="model",
            exist_ok=True,
            private=self.is_private,
        )

        # 一時フォルダの内容をまるごとアップロード
        self.api.upload_folder(
            folder_path=str(self.stage_dir),
            repo_id=self.hf_repo_id,
            repo_type="model",
            commit_message="Upload LoRA adapter (README written by author)",
        )

        print("✅ アップロードが正常に完了しました。")
        print(f"URL: https://huggingface.co/{self.hf_repo_id}")
