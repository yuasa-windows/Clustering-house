# 動作環境：VAR

import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime
import jpholiday


import warnings
warnings.filterwarnings("ignore")

# 作業ディレクトリの設定
try:
    os.chdir('H:/マイドライブ/03_code_test/clustering-house_trial')
    data_path_header = 'G:/マイドライブ/01_研究/02_円山町/1_データ前処理'
except FileNotFoundError:
    os.chdir('G:/マイドライブ/03_code_test/clustering-house_trial')
    data_path_header = 'H:/マイドライブ/01_研究/02_円山町/1_データ前処理'
print("Current Working Directory: ", os.getcwd())

# カスタムライブラリのパスを追加
sys.path.append(data_path_header)
from column_translation import column_translation_dict

# --------------------------------------------------------------------------------------
# データ読み込み関数
def load_data(house_num, start_date, end_date, col_list_road):
    df = pd.read_csv(os.path.join(data_path_header, f'12_一括処理後データ/大林新星和不動産/{house_num}号地/{house_num}_30Min.csv'), encoding='utf-8')
    df = df.rename(columns=column_translation_dict)
    df["time"] = pd.to_datetime(df["time"])
    df = df[(df["time"] >= start_date) & (df["time"] <= end_date)]
    df_data = df[['time'] + col_list_road]
    df_data.set_index('time', inplace=True)

    # 状態データの読み込み
    df_state = pd.DataFrame()
    col_remove = []
    for column in col_list_road:
        try:
            file_path_list = glob.glob(f'../GMM-HMM_Trial/output_HMM/{house_num}号地/{column}/*_mode.csv')
        except FileNotFoundError:
            print(f'No mode file found for {column} in house {house_num}. Skipping.')
            col_remove.append(column)
            continue

        if not file_path_list:
            print(f'No mode file found for {column} in house {house_num}.')
            col_remove.append(column)
            continue

        df_mode = pd.read_csv(file_path_list[-1], encoding='utf-8', index_col='time', parse_dates=['time'], usecols=['time', 'mode'])
        df_mode = df_mode.rename(columns={'mode': f'{column}_state'})
        if df_state.empty:
            df_state = df_mode
        else:
            df_state = pd.merge(df_state, df_mode, left_index=True, right_index=True)

    df_data = pd.merge(df_data, df_state, left_index=True, right_index=True)
    for col in col_remove:
        col_list_road.remove(col)
        print(f'Removed {col} from analysis due to missing state data.')
    return df_data, col_list_road

# 閾値の読み込み関数
def load_thresholds(house_num, col_list):
    try:
        file_path_list = glob.glob(f'../GMM-HMM_Trial/output_HMM/{house_num}号地/{col_list[0]}/*_result.csv')
    except FileNotFoundError:
        print(f'No result file found for {col_list[0]} in house {house_num}.')
        return None
    thresholds_csv = pd.read_csv(file_path_list[-1], encoding='utf-8')
    return thresholds_csv

# --------------------------------------------------------------------------------------
# 指標計算関数群
# 1. 合計消費量
def calc_mean_consumption(df_data, column):
    mean_annual = df_data[column].mean()
    mean_monthly = df_data[column].resample('M').mean()
    mean_monthly.index = mean_monthly.index.to_period('M')
    return round(mean_annual, 2), mean_monthly.round(2)

# 2. 時刻別消費量（0-6, 6-12, 12-18, 18-24）
def calc_time_bin_consumption(df_data, column, hours_per_bin=6):
    # 3時間区間に分類
    df_data['hour_bin'] = (df_data.index.hour // hours_per_bin) * hours_per_bin
    monthly_time_bin = df_data.groupby([df_data.index.to_period('M'), 'hour_bin'])[column].mean().unstack(level=1)
    monthly_time_bin.columns = [f'{h}-{h+hours_per_bin}' for h in monthly_time_bin.columns]
    return monthly_time_bin.round(2)

# 3. PCR（ピーク消費比率）
def calc_pcr(df_data, column):
    daily = df_data[column].resample('D')
    daily_mean = daily.mean()
    daily_max = daily.max()
    pcr_daily = (daily_max / daily_mean).replace([np.inf, -np.inf], np.nan)
    pcr_monthly = pcr_daily.resample('M').mean()
    pcr_monthly.index = pcr_monthly.index.to_period('M')
    return pcr_monthly.round(2)

# 4. 日夜消費量比率
def calc_day_night_ratio(df_data, column):
    def day_night(hour):
        return 'day' if 6 <= hour < 18 else 'night'
    df_data['day_night'] = df_data.index.hour.map(day_night)
    monthly_sum = df_data.groupby([df_data.index.to_period('M'), 'day_night'])[column].sum().unstack()
    monthly_sum['day_night_ratio'] = monthly_sum['day'] / monthly_sum['night']
    return monthly_sum['day_night_ratio'].round(2)

# --- 5. 平日・休日消費量比率（祝日対応版） ---
def calc_weekday_weekend_ratio(df_data, column):
    df_data['weekday'] = df_data.index.weekday
    df_data['is_holiday'] = df_data.index.to_series().apply(lambda x: jpholiday.is_holiday(x))
    df_data['day_type'] = df_data.apply(
        lambda row: 'weekend' if row['weekday'] >= 5 or row['is_holiday'] else 'weekday', axis=1
    )
    monthly_sum = df_data.groupby([df_data.index.to_period('M'), 'day_type'])[column].sum().unstack()
    monthly_sum['weekday_weekend_ratio'] = monthly_sum['weekday'] / monthly_sum['weekend']
    return monthly_sum['weekday_weekend_ratio'].round(2)

# --- 6. 稼働時間に対する消費量 + 時刻別稼働確率 ---
def calc_consumption_per_active_hour(df_data, column, hours_per_bin=6):
    # 稼働判定
    df_data['is_active'] = df_data[f'{column}_state'] >= 2
    monthly_active_hours = df_data.groupby(df_data.index.to_period('M'))['is_active'].sum() * 0.5
    active_df = df_data[df_data['is_active']].copy()
    monthly_active_consumption = active_df.groupby(active_df.index.to_period('M'))[column].sum()
    monthly_ratio = monthly_active_consumption / monthly_active_hours
    # --- 月別・6時間区間稼働確率 ---
    df_data['month'] = df_data.index.to_period('M')
    df_data['hour_bin'] = (df_data.index.hour // hours_per_bin) * hours_per_bin
    grouped = df_data.groupby(['month', 'hour_bin'])
    active_counts = grouped['is_active'].sum()
    total_counts = grouped['is_active'].count()
    hourly_active_probability = (active_counts / total_counts).unstack(level=1).round(2)
    hourly_active_probability.columns = [f'{h}-{h+hours_per_bin}' for h in hourly_active_probability.columns]

    return monthly_active_hours, monthly_ratio.round(2), hourly_active_probability

# --- 7. 世帯人数 ---
def get_household_size(house_num):
    num_household_dict = {
        80	: 3,
        81	: 6,
        82	: 3,
        83	: 4,
        115	: 3,
        117	: 4,
        118	: 4,
        120	: 3,
        121	: 2,
        124	: 4,
        125	: 4,
        126	: 3,
        127	: 3,
        147	: 4,
        148	: 4,
        150	: 4,
        152	: 6,
        155	: 5,
        156	: 3,
        157	: 2,
        84	: 4,
        92	: 4,
        94	: 4,
        116	: 4,
        119	: 4,
        149	: 2,
        154	: 4,
        158	: 3,
        160	: 3,
        171	: 3,
        172	: 4,
    }
    return num_household_dict.get(house_num, None)


# --------------------------------------------------------------------------------------
# 月別メトリクスのDataFrame組み立て関数
def build_monthly_metrics_df(
    df_data, column, metrics, house_num,
    time_bin_prefix='time_bin_',
    hap_prefix='hap_',
    rename_map=None
):

    if not metrics:
        raise ValueError("metrics は1つ以上指定してください。")

    # 1) ここで必要な計算だけ一度だけ実行
    need_total = 'total_consumption' in metrics
    need_pcr   = 'PCR' in metrics
    need_dnr   = 'day_night_ratio' in metrics
    need_wwr   = 'weekday_weekend_ratio' in metrics
    need_time  = 'time_bin' in metrics
    need_cah   = any(m in metrics for m in ['active_hours','consumption_per_active_hour','active_probability'])
    need_household = 'household_size' in metrics

    if need_total:
        total_annual, total_monthly = calc_mean_consumption(df_data, column)
    if need_pcr:
        pcr_monthly = calc_pcr(df_data, column)
    if need_dnr:
        day_night_ratio = calc_day_night_ratio(df_data, column)
    if need_wwr:
        weekday_weekend_ratio = calc_weekday_weekend_ratio(df_data, column)
    if need_time:
        time_bin_monthly = calc_time_bin_consumption(df_data, column, hours_per_bin=6)
    if need_cah:
        active_hours, consumption_per_active_hour, hourly_active_probability = \
            calc_consumption_per_active_hour(df_data, column, hours_per_bin=6)
    if need_household:
        household_size = get_household_size(house_num)

    # 2) 縦（Series/1列DF）として積むもの
    series_parts = []
    if 'total_consumption' in metrics:
        series_parts.append(('total_consumption', total_monthly))
    if 'PCR' in metrics:
        series_parts.append(('PCR', pcr_monthly))
    if 'day_night_ratio' in metrics:
        series_parts.append(('day_night_ratio', day_night_ratio))
    if 'weekday_weekend_ratio' in metrics:
        series_parts.append(('weekday_weekend_ratio', weekday_weekend_ratio))
    if 'active_hours' in metrics:
        # s = active_hours if isinstance(active_hours, pd.Series) else active_hours.squeeze()
        series_parts.append(('active_hours', active_hours))
    if 'consumption_per_active_hour' in metrics:
        # s = (consumption_per_active_hour if isinstance(consumption_per_active_hour, pd.Series) else consumption_per_active_hour.squeeze())
        series_parts.append(('consumption_per_active_hour', consumption_per_active_hour))

    # 3) まず縦を結合
    df_monthly_result = pd.concat(
        [s.rename(name) for name, s in series_parts],
        axis=1
    ) if series_parts else pd.DataFrame()

    # 4) 横（複数列DF）を順次 join（列名にプレフィックス）
    # time_bin
    if 'time_bin' in metrics:
        tb = time_bin_monthly.copy()
        if not isinstance(tb, pd.DataFrame):
            # 念のため Series の場合は1列DF化
            tb = tb.to_frame()
        tb.columns = [f'{time_bin_prefix}{c}' for c in tb.columns]
        df_monthly_result = df_monthly_result.join(tb, how='outer')

    # hourly_active_probability（横持ちDF想定に対応）
    if 'active_probability' in metrics:
        hap = hourly_active_probability
        if isinstance(hap, pd.DataFrame) and hap.shape[1] > 1:
            hap_df = hap.copy()
            hap_df.columns = [f'{hap_prefix}{c}' for c in hap_df.columns]
            df_monthly_result = df_monthly_result.join(hap_df, how='outer')
        # else:
        #     s = hap if isinstance(hap, pd.Series) else hap.squeeze()
        #     df_monthly_result = df_monthly_result.join(s.rename('active_probability'), how='outer')

    # 5) 列順を metrics の指定順に合わせる（横持ちはプレフィックスでまとめて抽出）
    ordered_cols = []
    for key in metrics:
        if key == 'time_bin':
            ordered_cols += [c for c in df_monthly_result.columns if c.startswith(time_bin_prefix)]
        elif key == 'active_probability':
            # 横持ち（プレフィックス付き）と縦持ち（1列）の両対応
            hap_cols = [c for c in df_monthly_result.columns if c.startswith(hap_prefix)]
            if hap_cols:
                ordered_cols += hap_cols
            elif 'active_probability' in df_monthly_result.columns:
                ordered_cols.append('active_probability')
        else:
            if key in df_monthly_result.columns:
                ordered_cols.append(key)

    if ordered_cols:
        df_monthly_result = df_monthly_result[ordered_cols]

    if 'household_size' in metrics:
        df_monthly_result['household_size'] = household_size

    # 6) 任意のリネーム
    if rename_map:
        df_monthly_result = df_monthly_result.rename(columns=rename_map)

    return df_monthly_result

# --------------------------------------------------------------------------------------
# メイン処理
target_dir = Path('../GMM-HMM_Trial/output_HMM')
dirs = [
    p.name.removesuffix('号地')
    for p in target_dir.iterdir()
    if p.is_dir() and p.name.endswith('号地')
]
house_list = sorted(dirs, key=lambda x: int(x))
# house_list = [83, 124]
start_date = '2024-04-01 00:00:00'
end_date = '2025-03-30 23:30:00'
col_list_origin = ['electric_demand', 'LD', 'kitchen', 'bedroom', 'bathroom', 'washing_machine', 'dishwasher']

for house_num in house_list:
    print(f"\n=== {house_num}号地 の分析 ===")
    df_data, col_list = load_data(house_num, start_date, end_date, col_list_origin.copy())
    # thresholds_csv = load_thresholds(house_num, col_list)
    print(f"分析対象の列: {col_list}")
    for column in col_list:
        print(f"\t--- {column} の分析結果 ---")

        # ここで出力したい指標だけ選ぶ
        selected_metrics = [
            'total_consumption',
            'PCR',
            'day_night_ratio',
            'weekday_weekend_ratio',
            'active_hours',
            'consumption_per_active_hour',
            'active_probability',
            'time_bin',
            'household_size',
        ]

        rename_map = {
            'total_consumption': 'mean_consumption',
            'day_night_ratio': 'day_night_ratio',
            'weekday_weekend_ratio': 'weekday_weekend_ratio',
            'active_hours': 'active_hours',
            'consumption_per_active_hour': 'consumption_per_active_hour',
            'active_probability': 'active_probability',
        }

        df_monthly_result = build_monthly_metrics_df(
            df_data, column,
            metrics=selected_metrics,
            house_num=house_num,
            time_bin_prefix='time_bin_',
            hap_prefix='act-pro_',
            rename_map=rename_map
        )
        # CSVに出力
        os.makedirs(f'./output_feature', exist_ok=True)
        df_monthly_result.to_csv(f'./output_feature/{house_num}_{column}_energy_metrics.csv', index_label='month')

    print("\ncomplete monthly metrics output to CSV.")

