# 動作環境：VAR

import os
import sys
import glob
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
def load_data(house_num, start_date, end_date, col_list):
    df = pd.read_csv(os.path.join(data_path_header, f'12_一括処理後データ/大林新星和不動産/{house_num}号地/{house_num}_30Min.csv'), encoding='utf-8')
    df = df.rename(columns=column_translation_dict)
    df["time"] = pd.to_datetime(df["time"])
    df = df[(df["time"] >= start_date) & (df["time"] <= end_date)]
    df_data = df[['time'] + col_list]
    df_data.set_index('time', inplace=True)

    # 状態データの読み込み
    df_state = pd.DataFrame()
    col_remove = []
    for column in col_list:
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
        col_list.remove(col)
        print(f'Removed {col} from analysis due to missing state data.')
    return df_data, col_list

# 閾値の読み込み関数
def load_thresholds(house_num, col_list):
    file_path_list = glob.glob(f'../GMM-HMM_Trial/output_HMM/{house_num}号地/{col_list[0]}/*_result.csv')
    thresholds_csv = pd.read_csv(file_path_list[-1], encoding='utf-8')
    return thresholds_csv

# 1. 合計消費量
def calc_mean_consumption(df_data, column):
    mean_annual = df_data[column].mean()
    mean_monthly = df_data[column].resample('M').mean()
    mean_monthly.index = mean_monthly.index.to_period('M')
    return round(mean_annual, 2), mean_monthly.round(2)

# 2. 時刻別消費量（0-6, 6-12, 12-18, 18-24）
def calc_time_bin_consumption(df_data, column):
    def time_bin(hour):
        if 0 <= hour < 6:
            return '0-6'
        elif 6 <= hour < 12:
            return '6-12'
        elif 12 <= hour < 18:
            return '12-18'
        else:
            return '18-24'
    df_data['time_bin'] = df_data.index.hour.map(time_bin)
    time_bin_monthly = df_data.groupby([df_data.index.to_period('M'), 'time_bin'])[column].mean().unstack()
    return time_bin_monthly.round(2)

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

# --- 6. 稼働時間に対する消費量 ---
def calc_consumption_per_active_hour(df_data, column):
    df_data['is_active'] = df_data[f'{column}_state'] >= 2
    monthly_active_hours = df_data.groupby(df_data.index.to_period('M'))['is_active'].sum() * 0.5
    active_df = df_data[df_data['is_active']].copy()
    monthly_active_consumption = active_df.groupby(active_df.index.to_period('M'))[column].sum()
    monthly_ratio = monthly_active_consumption / monthly_active_hours
    return monthly_active_hours, monthly_ratio.round(2)

# --------------------------------------------------------------------------------------
# メイン処理
house_list = [125]
start_date = '2024-04-01 00:00:00'
end_date = '2025-03-30 23:30:00'
col_list_origin = ['electric_demand', 'LD', 'kitchen', 'bedroom', 'washing_machine', 'dishwasher']

for house_num in house_list:
    print(f"\n=== {house_num}号地 の分析 ===")
    df_data, col_list = load_data(house_num, start_date, end_date, col_list_origin)
    thresholds_csv = load_thresholds(house_num, col_list)
    for column in col_list:
        print(f"\t--- {column} の分析結果 ---")
        total_annual, total_monthly = calc_mean_consumption(df_data, column)
        time_bin_monthly = calc_time_bin_consumption(df_data, column)
        pcr_monthly = calc_pcr(df_data, column)
        day_night_ratio = calc_day_night_ratio(df_data, column)
        weekday_weekend_ratio = calc_weekday_weekend_ratio(df_data, column)
        active_hours, consumption_per_active_hour = calc_consumption_per_active_hour(df_data, column)

        # --- 結果表示 ---
        print("\t1. 平均消費量（年間）:", total_annual)

        # 月別指標を1つのDataFrameにまとめる
        df_monthly_result = pd.DataFrame({
            'total_consumption': total_monthly,
            'PCR': pcr_monthly,
            'day_night_ratio': day_night_ratio,
            'weekday_weekend_ratio': weekday_weekend_ratio,
            'consumption_per_active_hour': consumption_per_active_hour
        })
        # 時刻別消費量（月平均）は別DataFrameなので、列名を整理して結合
        time_bin_monthly.columns = [f'time_bin_{col}' for col in time_bin_monthly.columns]
        df_monthly_result = df_monthly_result.join(time_bin_monthly)
        # CSVに出力
        os.makedirs(f'./output_feature', exist_ok=True)
        df_monthly_result.to_csv(f'./output_feature/{house_num}_{column}_energy_metrics.csv', index_label='month')

    print("\ncomplete monthly metrics output to CSV.")

