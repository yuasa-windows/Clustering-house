import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import jpholiday
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# グローバル設定
# ============================================================================

# fiscal_year を 2020〜2024 の間で変更して使う想定
FISCAL_YEARS = range(2021, 2025)

# HMM 出力のルートディレクトリs
HMM_BASE_DIR = Path("../GMM-HMM_Trial")

# 時間帯ラベル
TIME_BANDS = [
    "late_night",     # 0–4
    "early_morning",  # 4–6
    "morning",        # 6–8
    "daytime",        # 8–16
    "evening",        # 16–18
    "night",          # 18–24
]

# 作業ディレクトリとデータパス設定
try:
    os.chdir("H:/マイドライブ/03_code_test/clustering-house_trial")
    data_path_header = "G:/マイドライブ/01_研究/02_円山町/1_データ前処理"
except FileNotFoundError:
    os.chdir("G:/マイドライブ/03_code_test/clustering-house_trial")
    data_path_header = "H:/マイドライブ/01_研究/02_円山町/1_データ前処理"

print("Current Working Directory: ", os.getcwd())

# カスタムライブラリのパスを追加
sys.path.append(data_path_header)
from column_translation import column_translation_dict


# ============================================================================
# ヘルパ関数
# ============================================================================

def get_hmm_output_dir(fiscal_year: int) -> Path:
    """
    HMM の出力ディレクトリパスを返す。

    Parameters
    ----------
    fiscal_year : int
        対象とする年度（例: 2020, 2021, ...）。

    Returns
    -------
    Path
        output_HMM_fyXXXX ディレクトリのパス。
    """
    return HMM_BASE_DIR / f"output_HMM_fy{fiscal_year}"


def load_data(house_num: int, start_date: str, end_date: str,
              col_list_road: list, fiscal_year: int):
    """
    電力消費データと状態データを読み込み、結合して返す。

    Parameters
    ----------
    house_num : int
        号地番号。
    start_date : str
        抽出開始日時（例: '2020-04-01 00:00:00'）。
    end_date : str
        抽出終了日時（例: '2021-03-31 23:30:00'）。
    col_list_road : list of str
        読み込む計測項目のリスト。
    fiscal_year : int
        使用する HMM 出力の年度。

    Returns
    -------
    df_data : pandas.DataFrame
        時系列インデックスを持つ電力データ＋状態データ。
    col_list_road : list of str
        状態データが存在した計測項目のリスト。
    """
    # 電力消費データの読み込み
    csv_path = os.path.join(
        data_path_header,
        f"12_一括処理後データ/大林新星和不動産/{house_num}号地/{house_num}_30Min.csv",
    )
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df.rename(columns=column_translation_dict)
    df["time"] = pd.to_datetime(df["time"])
    df = df[(df["time"] >= start_date) & (df["time"] <= end_date)]
    df_data = df[["time"] + col_list_road].copy()
    df_data.set_index("time", inplace=True)

    # 状態データの読み込み
    df_state = pd.DataFrame()
    col_remove = []

    hmm_dir = get_hmm_output_dir(fiscal_year)

    for column in col_list_road:
        # 状態ファイルのパスパターン作成
        pattern = hmm_dir / f"{house_num}号地" / column / "*_mode.csv"
        file_path_list = glob.glob(str(pattern))

        if not file_path_list:
            print(f"No mode file found for {column} in house {house_num}.")
            col_remove.append(column)
            continue

        # 最後のファイルを使用（最新と仮定）
        df_mode = pd.read_csv(
            file_path_list[-1],
            encoding="utf-8",
            index_col="time",
            parse_dates=["time"],
            usecols=["time", "mode"],
        )
        df_mode = df_mode.rename(columns={"mode": f"{column}_state"})

        if df_state.empty:
            df_state = df_mode
        else:
            df_state = pd.merge(df_state, df_mode, left_index=True, right_index=True)

    # 電力データと状態データを結合
    df_data = pd.merge(df_data, df_state, left_index=True, right_index=True)

    # 状態データがなかった列はリストから除外
    for col in col_remove:
        col_list_road.remove(col)
        print(f"Removed {col} from analysis due to missing state data.")

    return df_data, col_list_road


def add_time_band(df: pd.DataFrame) -> pd.DataFrame:
    """
    各タイムスタンプに 6 区分の time_band 列を付与する。

    - late_night    : 0–4
    - early_morning : 4–6
    - morning       : 6–8
    - daytime       : 8–16
    - evening       : 16–18
    - night         : 18–24

    Parameters
    ----------
    df : pandas.DataFrame
        DatetimeIndex を持つ DataFrame。

    Returns
    -------
    pandas.DataFrame
        time_band 列が追加された DataFrame。
    """
    if "time_band" in df.columns:
        return df

    df = df.copy()
    hour = df.index.hour

    cond_list = [
        (hour >= 0) & (hour < 4),
        (hour >= 4) & (hour < 6),
        (hour >= 6) & (hour < 8),
        (hour >= 8) & (hour < 16),
        (hour >= 16) & (hour < 18),
        (hour >= 18) & (hour < 24),
    ]

    df["time_band"] = np.select(cond_list, TIME_BANDS)
    return df


def add_half_hour_bin(df: pd.DataFrame) -> pd.DataFrame:
    """
    30分ごとのビンID（0〜47）を half_hour_bin 列として付与する。

    Parameters
    ----------
    df : pandas.DataFrame
        DatetimeIndex を持つ DataFrame。

    Returns
    -------
    pandas.DataFrame
        half_hour_bin 列が追加された DataFrame。
    """
    if "half_hour_bin" not in df.columns:
        df = df.copy()
        df["half_hour_bin"] = df.index.hour * 2 + (df.index.minute // 30)
    return df


def add_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    平日/休日・祝日・PV時間帯フラグを付与する。

    day_type:
        weekday or weekend（土日 or 祝日）
    is_pv_time:
        8:00〜15:30 を True

    Parameters
    ----------
    df : pandas.DataFrame
        DatetimeIndex を持つ DataFrame。

    Returns
    -------
    pandas.DataFrame
        weekday, is_holiday, day_type, is_pv_time 列が追加された DataFrame。
    """
    df = df.copy()

    if "weekday" not in df.columns:
        df["weekday"] = df.index.weekday
    if "is_holiday" not in df.columns:
        df["is_holiday"] = df.index.to_series().apply(lambda d: jpholiday.is_holiday(d) is not None).astype(bool)
    if "day_type" not in df.columns:
        df["day_type"] = np.where(
            (df["weekday"] >= 5) | (df["is_holiday"]), "weekend", "weekday"
        )
    if "is_pv_time" not in df.columns:
        df["is_pv_time"] = (df.index.hour >= 8) & (df.index.hour <= 15)

    return df


def process_data(house: int, start_date: str, end_date: str,
                 col_list: list, fiscal_year: int):
    """
    1世帯分のデータを読み込み、time_band・カレンダー情報・half_hour_bin を付与する。

    Parameters
    ----------
    house : int
        号地番号。
    start_date : str
        抽出開始日時。
    end_date : str
        抽出終了日時。
    col_list : list of str
        対象とする計測項目リスト。
    fiscal_year : int
        HMM 出力の年度。

    Returns
    -------
    df : pandas.DataFrame
        加工済みの DataFrame。
    col_list : list of str
        状態データが存在した計測項目のリスト。
    """
    df, col_list = load_data(house, start_date, end_date, col_list, fiscal_year)

    if df.empty:
        # 空なら空 DataFrame を返す or None にするなど
        return pd.DataFrame(), col_list

    df = add_time_band(df)
    df = add_calendar_flags(df)
    df = add_half_hour_bin(df)
    return df, col_list


# ============================================================================
# 特徴量計算関数群
# ============================================================================

def calc_timeband_consumption(df_data: pd.DataFrame, column: str) -> pd.Series:
    """
    各 time_band における平均消費量を計算する。

    Parameters
    ----------
    df_data : pandas.DataFrame
        対象期間の時系列データ。
    column : str
        計測項目名。

    Returns
    -------
    pandas.Series
        index = TIME_BANDS, value = 各時間帯の平均消費量。
    """
    df = add_time_band(df_data)
    grouped = df.groupby("time_band")[column].mean()
    return grouped.reindex(TIME_BANDS)


def calc_timeband_active_rate(df_data: pd.DataFrame, column: str) -> pd.Series:
    """
    各 time_band における稼働率（稼働TS数 / 全TS数）を計算する。

    Parameters
    ----------
    df_data : pandas.DataFrame
        対象期間の時系列データ。
    column : str
        計測項目名。

    Returns
    -------
    pandas.Series
        index = TIME_BANDS, value = 各時間帯の稼働率。
    """
    state_col = f"{column}_state"
    if state_col not in df_data.columns:
        raise ValueError(f"状態列 {state_col} が存在しません。")

    df = df_data.copy()
    df["is_active"] = df[state_col] >= 2
    df = add_time_band(df)

    grouped = df.groupby("time_band")["is_active"].mean()
    return grouped.reindex(TIME_BANDS)


def calc_timeband_on_transitions(df_data: pd.DataFrame, column: str) -> pd.Series:
    """
    各 time_band における ON 遷移回数（1→2以上）を計算する。

    Parameters
    ----------
    df_data : pandas.DataFrame
        対象期間の時系列データ。
    column : str
        計測項目名。

    Returns
    -------
    pandas.Series
        index = TIME_BANDS, value = 各時間帯の ON 遷移回数。
    """
    state_col = f"{column}_state"
    if state_col not in df_data.columns:
        raise ValueError(f"状態列 {state_col} が存在しません。")

    df = df_data.copy()
    prev_state = df[state_col].shift(1)
    df["is_on_transition"] = (prev_state == 1) & (df[state_col] >= 2)
    df = add_time_band(df)

    grouped = df[df["is_on_transition"]].groupby("time_band").size()
    return grouped.reindex(TIME_BANDS, fill_value=0)


def calc_continuous_run_mean(df_data: pd.DataFrame, column: str,
                             step_hours: float = 0.5) -> float:
    """
    state>=2 の連続稼働区間の長さ（時間）の平均値を計算する。

    Parameters
    ----------
    df_data : pandas.DataFrame
        対象期間の時系列データ。
    column : str
        計測項目名。
    step_hours : float, default 0.5
        1ステップあたりの時間（時間単位）。

    Returns
    -------
    float
        連続稼働時間の平均（時間）。稼働がない場合や欠損がある場合は NaN。
    """
    state_col = f"{column}_state"
    if state_col not in df_data.columns:
        raise ValueError(f"状態列 {state_col} が存在しません。")

    active = df_data[state_col] >= 2
    if active.isna().any():
        return np.nan

    group_id = (active != active.shift(1)).cumsum()
    tmp = pd.DataFrame({"active": active, "group_id": group_id})
    runs = tmp[tmp["active"]].groupby("group_id")["active"].size()

    if runs.empty:
        return np.nan

    durations_hours = runs * step_hours
    return float(durations_hours.mean())


def calc_cv(df_data: pd.DataFrame, column: str) -> float:
    """
    期間全体の CV（変動係数）を計算する。

    CV = 標準偏差 / 平均

    Parameters
    ----------
    df_data : pandas.DataFrame
        対象期間の時系列データ。
    column : str
        計測項目名。

    Returns
    -------
    float
        CV。平均が 0 以下または欠損を含む場合は NaN。
    """
    series = df_data[column]
    if series.isna().any():
        return np.nan

    mean = series.mean()
    if mean <= 0:
        return np.nan

    std = series.std(ddof=0)
    return float(std / mean)


def calc_timeband_distribution_and_entropy(df_data: pd.DataFrame, column: str,
                                           log_base: float = np.e):
    """
    30分ビンごとの消費量分布 p を計算し、エントロピーと一様分布との KL を求める。

    Parameters
    ----------
    df_data : pandas.DataFrame
        対象期間の時系列データ。
    column : str
        計測項目名。
    log_base : float, default np.e
        対数の底。

    Returns
    -------
    p : pandas.Series
        index = 0..47, value = 各ビンの割合。
    entropy : float
        エントロピー H(p)。
    kl_uniform : float
        一様分布 q_i = 1/48 との KL ダイバージェンス。
    """
    df = add_half_hour_bin(df_data)
    series = df[column]

    if series.isna().any():
        p = pd.Series([np.nan] * 48, index=range(48))
        return p, np.nan, np.nan

    s_i = df.groupby("half_hour_bin")[column].sum()
    s_i = s_i.reindex(range(48), fill_value=0.0)
    S = s_i.sum()
    if S <= 0:
        p = pd.Series([np.nan] * 48, index=range(48))
        return p, np.nan, np.nan

    p = s_i / S
    p_nonzero = p[p > 0]

    if log_base == np.e:
        logs = np.log(p_nonzero)
    elif log_base == 2:
        logs = np.log2(p_nonzero)
    elif log_base == 10:
        logs = np.log10(p_nonzero)
    else:
        logs = np.log(p_nonzero) / np.log(log_base)

    entropy = -float((p_nonzero * logs).sum())

    if log_base == np.e:
        kl_uniform = float(np.log(48) - entropy)
    else:
        q = 1.0 / 48.0
        if log_base == 2:
            logs_ratio = np.log2(p_nonzero / q)
        elif log_base == 10:
            logs_ratio = np.log10(p_nonzero / q)
        else:
            logs_ratio = np.log(p_nonzero / q) / np.log(log_base)
        kl_uniform = float((p_nonzero * logs_ratio).sum())

    return p, entropy, kl_uniform


def calc_weekday_weekend_ratio_total(df_data: pd.DataFrame, column: str) -> float:
    """
    平日と休日（祝日含む）の総消費量の比率を計算する。

    ratio = S_weekday / S_weekend

    Parameters
    ----------
    df_data : pandas.DataFrame
        対象期間の時系列データ。
    column : str
        計測項目名。

    Returns
    -------
    float
        平日/休日比。片方しかない、または weekend=0 の場合は NaN。
    """
    df = add_calendar_flags(df_data)
    grouped = df.groupby("day_type")[column].sum()

    if "weekday" not in grouped.index or "weekend" not in grouped.index:
        return np.nan

    s_weekday = grouped["weekday"]
    s_weekend = grouped["weekend"]
    if s_weekend == 0:
        return np.nan

    return float(s_weekday / s_weekend)


def calc_pv_features(df_data: pd.DataFrame, column: str,
                     pv_const_MJ: float = 300.0, mj_per_kwh: float = 9.76):
    """
    PV 時間帯の消費量から PV 関連指標を計算する。

    1) pv_share（S_pv / S_all）※CSVには出力しない
    2) pv_ratio_to_const = S_pv(kWh) / PV_const(kWh)

    Parameters
    ----------
    df_data : pandas.DataFrame
        対象期間の時系列データ。
    column : str
        計測項目名。
    pv_const_MJ : float, default 300.0
        PV 発電量の定数（MJ）。
    mj_per_kwh : float, default 9.76
        1kWh あたりの MJ。

    Returns
    -------
    pv_share : float
        消費量に対する PV 時間帯の割合。
    pv_ratio_to_const : float
        PV_const に対する PV 時間帯消費量の比。
    """
    df = add_time_band(df_data)
    series = df[column]

    if series.isna().any():
        return np.nan, np.nan

    s_all = series.sum()
    s_pv = series[df["time_band"] == "daytime"].sum()
    pv_share = np.nan if s_all <= 0 else float(s_pv / s_all)

    pv_const_kwh = pv_const_MJ / mj_per_kwh
    pv_ratio_to_const = np.nan if pv_const_kwh <= 0 else float(s_pv / pv_const_kwh)

    return pv_share, pv_ratio_to_const


def calc_mean_electric_demand(df_data: pd.DataFrame) -> float:
    """
    electric_demand の平均値を計算する。

    Parameters
    ----------
    df_data : pandas.DataFrame
        対象期間の時系列データ。

    Returns
    -------
    float
        electric_demand の平均。欠損がある場合は NaN。
    """
    if "electric_demand" not in df_data.columns:
        raise ValueError("electric_demand 列が存在しません。")

    series = df_data["electric_demand"]
    if series.isna().any():
        return np.nan

    return float(series.mean())


def compute_all_features(df_data: pd.DataFrame, column: str,
                         pv_const_MJ: float = 300.0, mj_per_kwh: float = 9.76):
    """
    1〜9 の特徴量をまとめて計算する。

    Parameters
    ----------
    df_data : pandas.DataFrame
        対象期間の時系列データ。
    column : str
        計測項目名。
    pv_const_MJ : float, default 300.0
        PV 発電量の定数（MJ）。
    mj_per_kwh : float, default 9.76
        1kWh あたりの MJ。

    Returns
    -------
    dict
        各特徴量名をキーとする辞書。
    """
    timeband_consumption = calc_timeband_consumption(df_data, column)
    timeband_active_rate = calc_timeband_active_rate(df_data, column)
    timeband_on_counts = calc_timeband_on_transitions(df_data, column)
    cont_run_mean = calc_continuous_run_mean(df_data, column)
    cv = calc_cv(df_data, column)
    p_dist, entropy, kl_uniform = calc_timeband_distribution_and_entropy(df_data, column)
    weekday_weekend_ratio = calc_weekday_weekend_ratio_total(df_data, column)
    pv_share, pv_ratio_to_const = calc_pv_features(df_data, column, pv_const_MJ, mj_per_kwh)
    mean_elec_demand = calc_mean_electric_demand(df_data)

    return {
        "timeband_consumption": timeband_consumption,
        "timeband_active_rate": timeband_active_rate,
        "timeband_on_counts": timeband_on_counts,
        "cont_run_mean": cont_run_mean,
        "cv": cv,
        "timeband_distribution": p_dist,
        "entropy": entropy,
        "kl_uniform": kl_uniform,
        "weekday_weekend_ratio": weekday_weekend_ratio,
        "pv_share": pv_share,
        "pv_ratio_to_const": pv_ratio_to_const,
        "mean_elec_demand": mean_elec_demand,
    }


# ============================================================================
# CSV 出力用関数
# ============================================================================

def build_feature_table(df_data: pd.DataFrame, column: str,
                        agg: str = "monthly") -> pd.DataFrame:
    """
    1世帯分の df_data から、指定項目の月次 or 週次特徴量テーブルを生成する。

    Parameters
    ----------
    df_data : pandas.DataFrame
        1世帯分の時系列データ。
    column : str
        計測項目名。
    agg : {'monthly', 'weekly'}, default 'monthly'
        集約単位。

    Returns
    -------
    pandas.DataFrame
        各期間（month or week）ごとの特徴量を1行にもつ DataFrame。
    """
    df = df_data.copy()

    if agg == "monthly":
        df["period"] = df.index.to_period("M")
        period_col_name = "month"
    elif agg == "weekly":
        df["period"] = df.index.to_period("W-MON")
        period_col_name = "week"
    else:
        raise ValueError("agg must be 'monthly' or 'weekly'.")

    rows = []

    for period, df_period in df.groupby("period"):
        feats = compute_all_features(df_period, column)
        row = {}

        # 期間ラベル
        if agg == "monthly":
            row[period_col_name] = period.strftime("%Y-%m")
        else:
            start_date = period.start_time.date()
            row[period_col_name] = start_date.isoformat()

        # 1. 時間別消費量
        for band in TIME_BANDS:
            row[f"consumption_{band}"] = float(
                feats["timeband_consumption"].get(band, np.nan)
            )
        # 2. 時間別稼働率
        for band in TIME_BANDS:
            row[f"active_rate_{band}"] = float(
                feats["timeband_active_rate"].get(band, np.nan)
            )
        # 3. ON/OFF 遷移回数
        for band in TIME_BANDS:
            row[f"on_count_{band}"] = int(
                feats["timeband_on_counts"].get(band, 0)
            )
        # 4. 連続稼働時間
        row["cont_run_mean"] = feats["cont_run_mean"]
        # 5. CV
        row["cv"] = feats["cv"]
        # 6. KL（entropy, p_dist は CSV に出さない）
        row["kl_uniform_timeband"] = feats["kl_uniform"]
        # 7. 平日・休日比
        row["weekday_weekend_ratio"] = feats["weekday_weekend_ratio"]
        # 8. PV_const 比（pv_share は CSV に出さない）
        row["pv_ratio_to_const"] = feats["pv_ratio_to_const"]
        # 9. 目的変数
        row["mean_elec_demand"] = feats["mean_elec_demand"]

        rows.append(row)

    feature_df = pd.DataFrame(rows)
    feature_df.sort_values(by=period_col_name, inplace=True)
    return feature_df

def save_feature_csv_for_house(df_data: pd.DataFrame, house_id: int, items: list,
                               agg: str = "monthly",
                               output_root: str = "./output_feature"):
    """
    1世帯分の df_data と項目リストから、特徴量 CSV を出力する。

    ※年度をまたいで同じ CSV に追記する。
      例: ./output_feature/80/features_monthly_80_bathroom.csv に
          2020-04〜2025-03 の全期間を順次 append していく。

    Parameters
    ----------
    df_data : pandas.DataFrame
        1世帯分の時系列データ（ある年度の分）。
    house_id : int
        号地番号。
    items : list of str
        対象とする計測項目リスト。
    agg : {'monthly', 'weekly'}, default 'monthly'
        集約単位。
    output_root : str, default './output_feature'
        出力先ルートディレクトリ。

    Returns
    -------
    None
    """
    house_dir = os.path.join(output_root, str(house_id))
    os.makedirs(house_dir, exist_ok=True)

    for item in items:
        feature_df = build_feature_table(df_data, item, agg=agg)
        feature_df = feature_df.round(3)

        filename = f"features_{agg}_{house_id}_{item}.csv"
        out_path = os.path.join(house_dir, filename)

        # すでに同名ファイルがある場合 → 追記（ヘッダなし）
        # ない場合 → 新規作成（ヘッダあり）
        if os.path.exists(out_path):
            feature_df.to_csv(out_path, mode="a", header=False, index=False)
            print(f"Appended to existing file: {out_path}")
        else:
            feature_df.to_csv(out_path, index=False)
            print(f"Created new file: {out_path}")


# ============================================================================
# 使用例（スクリプトとして実行する場合）
# ============================================================================

if __name__ == "__main__":
    for fiscal_year in FISCAL_YEARS:
        target_dir = get_hmm_output_dir(fiscal_year)

        # 号地一覧の取得
        dirs = [
            p.name.removesuffix("号地")
            for p in target_dir.iterdir()
            if p.is_dir() and p.name.endswith("号地")
        ]
        house_list = sorted(dirs, key=lambda x: int(x))

        # 例として、fiscal_year の 1 年分を処理
        start_date = f"{fiscal_year}-04-01 00:00:00"
        end_date = f"{fiscal_year + 1}-03-31 23:30:00"

        col_list_origin = [
            "electric_demand",
            "LD",
            "kitchen",
            "bedroom",
            "bathroom",
            "washing_machine",
            "dishwasher",
        ]

        MIN_ROWS = 365 * 48 # 最低限必要な行数（1年分の30分間隔データ）

        for house_id in house_list:
            df_data, col_list = process_data(
                house_id, start_date, end_date, col_list_origin.copy(), fiscal_year
            )

            # データ不足ならスキップ
            if len(df_data) < MIN_ROWS:
                print(
                    f"[SKIP] house_id={house_id}: "
                    f"len(df_data)={len(df_data)} < {MIN_ROWS}"
                )
                continue  # 次の house_id へ

            # 月次特徴量 CSV
            save_feature_csv_for_house(
                df_data, house_id=house_id, items=col_list, agg="monthly"
            )

            # 週次特徴量 CSV
            save_feature_csv_for_house(
                df_data, house_id=house_id, items=col_list, agg="weekly"
            )
