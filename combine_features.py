import os
import sys
import glob
from pathlib import Path
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import rename_dict  # item_name_dict, feature_name_dict を想定

warnings.filterwarnings("ignore")


# =========================================================
# 作業ディレクトリ・パス設定まわり
# =========================================================
def setup_workdir() -> str:
    """
    作業ディレクトリを候補リストから設定し、
    column_translation 用の data_path_header を返す。
    """
    candidates = [
        (
            "H:/マイドライブ/03_code_test/clustering-house_trial",
            "G:/マイドライブ/01_研究/02_円山町/1_データ前処理",
        ),
        (
            "G:/マイドライブ/03_code_test/clustering-house_trial",
            "H:/マイドライブ/01_研究/02_円山町/1_データ前処理",
        ),
    ]

    for wd, data_path_header in candidates:
        try:
            os.chdir(wd)
            print("Current Working Directory:", os.getcwd())
            return data_path_header
        except FileNotFoundError:
            continue

    raise FileNotFoundError("想定したどの作業ディレクトリも存在しません。")


# =========================================================
# ユーティリティ関数
# =========================================================
def detect_date_column(df: pd.DataFrame, candidates=("date", "month", "week")) -> str:
    """
    与えられた DataFrame から日付列を推定して列名を返す。
    見つからなければエラー。
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"日付列 {candidates} が見つかりませんでした。columns={df.columns.tolist()}")


def parse_meta_from_filename(path: Path):
    """
    features_{agg}_{house_id}_{item}.csv というファイル名から
    agg, house_id, item を抽出する。
    """
    basename = path.name  # e.g. features_monthly_116_electric_demand.csv
    core = basename[len("features_"):-4]  # "monthly_116_electric_demand"
    parts = core.split("_")

    agg = parts[0]
    house_id = parts[1]
    item = "_".join(parts[2:])  # electric_demand / washing_machine など "_" 含むものに対応
    return agg, house_id, item


# =========================================================
# データ読み込み・結合
# =========================================================
def load_all_features(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    base_dir 配下の features_*.csv をすべて読み込み、
    縦結合した df_all と、メタ情報のみの meta_df を返す。
    """
    all_dfs = []
    meta_records = []

    # ./output_feature/*/features_*.csv をすべて拾う
    for fpath in base_dir.glob("*/features_*.csv"):
        agg, house_id, item = parse_meta_from_filename(fpath)

        df = pd.read_csv(fpath)

        # 日付列を統一
        date_col = detect_date_column(df)
        df = df.rename(columns={date_col: "date"})

        # メタ情報付与
        df["house_id"] = house_id
        df["agg"] = agg
        df["item"] = item

        all_dfs.append(df)
        meta_records.append({"agg": agg, "house_id": house_id, "item": item})

    if not all_dfs:
        raise RuntimeError(f"{base_dir} 以下に features_*.csv が見つかりませんでした。")

    df_all = pd.concat(all_dfs, ignore_index=True)
    meta_df = pd.DataFrame(meta_records).drop_duplicates().reset_index(drop=True)

    return df_all, meta_df


# =========================================================
# ワイド形式への変換
# =========================================================
def make_wide_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    縦持ちの df_all を house_id × agg × date のワイド形式に変換し、
    item/feature の名前を rename_dict に従って短縮する。
    さらに mean_elec_demand（mEDm）を item をまたいで 1 列に集約する。
    """
    # item を列方向に展開
    df_wide = df_all.pivot_table(
        index=["house_id", "agg", "date"],
        columns="item",
        aggfunc="first",  # 特徴量は重複しない前提
    )

    # MultiIndex(columns) → フラットな列名に変換
    df_wide.columns = [
        f"{rename_dict.item_name_dict.get(item, item)}_"
        f"{rename_dict.feature_name_dict.get(col, col)}"
        for col, item in df_wide.columns
    ]
    df_wide = df_wide.reset_index()

    # {item}_mEDm 列を 1 列に統合（mEDm）
    mean_cols = [c for c in df_wide.columns if c.endswith("_mEDm")]
    if mean_cols:
        df_wide["mEDm"] = df_wide[mean_cols].bfill(axis=1).iloc[:, 0]
        df_wide = df_wide.drop(columns=mean_cols)

    return df_wide

# =========================================================
# 特徴量の欠損状況チェック
# =========================================================
def show_nan_ratio(df: pd.DataFrame, name: str = "df") -> None:
    n_rows = len(df)
    n_nan_rows = df.isna().any(axis=1).sum()
    ratio = n_nan_rows / n_rows if n_rows > 0 else np.nan

    print(f"=== {name} ===")
    print(f"総行数: {n_rows}")
    print(f"欠損を1つ以上含む行数: {n_nan_rows}")
    print(f"欠損を含む行の割合: {ratio:.3%}")  # パーセンテージ表示

def show_nan_by_column(df: pd.DataFrame, name: str = "df") -> None:
    print(f"\n=== {name} 欠損率（列ごと） ===")
    na_ratio = df.isna().mean().sort_values(ascending=False)
    print(na_ratio[na_ratio > 0])  # 欠損がある列だけ表示


# =========================================================
# 特徴量の欠損状況チェック（デバッグ用）
# =========================================================
def debug_feature_coverage(df_all: pd.DataFrame, df_wide: pd.DataFrame) -> None:
    """
    item ごとの特徴量数と、pivot 後に欠けている列をチェックする。
    デバッグ用途なので、問題なければ呼ばなくても良い。
    """
    print("\n=== item ごとの特徴量数 ===")
    items = df_all["item"].unique()
    item_features: dict[str, set] = {}

    for it in items:
        df_sub = df_all[df_all["item"] == it]
        feature_cols = [
            c for c in df_sub.columns
            if c not in ["date", "house_id", "agg", "item"]
        ]
        item_features[it] = set(feature_cols)

    for it, feats in item_features.items():
        print(it, len(feats))

    print("\n=== item 間の特徴量差分 ===")
    for a, b in combinations(items, 2):
        print(f"\n--- {a} vs {b} ---")
        print("AにあってBにない:", item_features[a] - item_features[b])
        print("BにあってAにない:", item_features[b] - item_features[a])

    # 期待される全特徴量名（item × feature）
    expected = sorted(
        f"{it}_{col}"
        for it in items
        for col in df_all.columns
        if col not in ["date", "house_id", "agg", "item"]
    )

    # pivot 後に実際に存在する列（メタ列を除く）
    actual = sorted(
        c for c in df_wide.columns
        if c not in ["date", "house_id", "agg"]
    )

    missing = set(expected) - set(actual)

    print("\n=== pivot後に欠けている列 ===")
    print("欠けている列数:", len(missing))
    print("欠けている列名:", missing)

    for col in missing:
        item, feat = col.split("_", 1)
        df_sub = df_all[df_all["item"] == item]

        if feat in df_sub.columns:
            print(col, "存在する。しかし pivot で消えた → 全NaNの可能性")
        else:
            print(col, "存在しない → CSVに最初から無い、または読み込みで欠損")


# =========================================================
# メイン処理
# =========================================================
def main():
    # 作業ディレクトリ設定 & カスタムライブラリパス追加
    data_path_header = setup_workdir()
    sys.path.append(data_path_header)

    base_dir = Path("./output_feature")

    # 1. 入力 CSV から縦持ち df_all とメタ情報 meta_df を作る
    df_all, meta_df = load_all_features(base_dir)

    agg_list = sorted(meta_df["agg"].unique().tolist())
    house_id_list = sorted(meta_df["house_id"].unique().tolist())
    item_list = sorted(meta_df["item"].unique().tolist())

    print("\n=== 検出された種類 ===")
    print("agg:", agg_list)
    print("house_id:", len(house_id_list), house_id_list)
    print("item:", len(item_list), item_list)

    # 2. ワイド形式に変換
    df_wide = make_wide_features(df_all)

    # 3. monthly / weekly に分割して保存
    df_monthly = df_wide[df_wide["agg"] == "monthly"].copy()
    df_weekly = df_wide[df_wide["agg"] == "weekly"].copy()

    out_monthly = base_dir / "combined_monthly_features.csv"
    out_weekly = base_dir / "combined_weekly_features.csv"

    df_monthly.to_csv(out_monthly, index=False)
    df_weekly.to_csv(out_weekly, index=False)

    print("\n=== 出力結果 ===")
    print("月次データの形状:", df_monthly.shape, "->", out_monthly)
    print("週次データの形状:", df_weekly.shape, "->", out_weekly)

    # 欠損状況チェック
    show_nan_ratio(df_monthly, name="月次 df_monthly")
    show_nan_ratio(df_weekly, name="週次 df_weekly")
    # show_nan_by_column(df_monthly, name="月次 df_monthly")
    # show_nan_by_column(df_weekly, name="週次 df_weekly")

    # 4. デバッグ用（必要なときだけ有効化）
    # debug_feature_coverage(df_all, df_wide)


if __name__ == "__main__":
    main()
