"""
温度数据 & TR10B电机数据 CSV合并工具
=====================================
用法:
    python merge_csv_tool.py <温度CSV路径> <电机CSV路径> [输出CSV路径]

说明:
    - 自动解析两个文件的时间戳，取共同时间范围
    - 以电机数据为基准，用最近邻时间戳匹配温度数据（容差1秒）
    - 自动剔除温度数据中所有温度列同时为0的异常行
    - 输出合并后的CSV文件

示例:
    python merge_csv_tool.py EA3862934_Wen-Du-Tong-Ji-Shu-Ju.csv TR10B_data_20260331_102344-2.csv merged_output.csv
"""

import sys
import pandas as pd
import numpy as np
import os


def load_temperature_csv(filepath):
    """读取温度统计CSV（跳过第一行单位信息，第二行为表头）"""
    df = pd.read_csv(filepath, skiprows=1, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]
    # 第一列是时间戳
    df['timestamp'] = pd.to_datetime(df.iloc[:, 0].str.strip(), format='%Y/%m/%d %H:%M:%S.%f')
    return df


def load_motor_csv(filepath):
    """读取TR10B电机CSV"""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df.iloc[:, 0].str.strip(), format='%Y/%m/%d %H:%M:%S.%f')
    return df


def remove_outliers(df, data_cols):
    """
    剔除异常行：所有数据列同时为0（采集故障），
    或任一数据列超出 [Q0.01 - 3*IQR, Q0.99 + 3*IQR] 范围
    """
    # 1) 所有温度列同时为0 → 明显采集故障
    all_zero_mask = pd.Series(True, index=df.index)
    for col in data_cols:
        if col in df.columns:
            all_zero_mask &= (pd.to_numeric(df[col], errors='coerce') == 0)
    
    # 2) IQR 异常检测
    iqr_mask = pd.Series(False, index=df.index)
    for col in data_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            Q1 = vals.quantile(0.01)
            Q3 = vals.quantile(0.99)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            iqr_mask |= (vals < lower) | (vals > upper) | vals.isna()
    
    outlier_mask = all_zero_mask | iqr_mask
    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        print(f"  剔除 {n_outliers} 行异常数据")
    return df[~outlier_mask].copy()


def merge_data(temp_path, motor_path, output_path=None):
    """主合并逻辑"""
    # ---------- 加载数据 ----------
    print(f"[1/5] 读取温度文件: {temp_path}")
    df_temp = load_temperature_csv(temp_path)
    print(f"      {len(df_temp)} 行, 时间范围: {df_temp['timestamp'].min()} ~ {df_temp['timestamp'].max()}")

    print(f"[2/5] 读取电机文件: {motor_path}")
    df_motor = load_motor_csv(motor_path)
    print(f"      {len(df_motor)} 行, 时间范围: {df_motor['timestamp'].min()} ~ {df_motor['timestamp'].max()}")

    # ---------- 共同时间范围 ----------
    common_start = max(df_temp['timestamp'].min(), df_motor['timestamp'].min())
    common_end = min(df_temp['timestamp'].max(), df_motor['timestamp'].max())
    print(f"[3/5] 共同时间范围: {common_start} ~ {common_end}")

    if common_start >= common_end:
        print("错误: 两个文件没有重叠的时间范围!")
        sys.exit(1)

    df_temp = df_temp[(df_temp['timestamp'] >= common_start) & (df_temp['timestamp'] <= common_end)]
    df_motor = df_motor[(df_motor['timestamp'] >= common_start) & (df_motor['timestamp'] <= common_end)]
    print(f"      过滤后: 温度 {len(df_temp)} 行, 电机 {len(df_motor)} 行")

    # ---------- 剔除异常值 ----------
    print("[4/5] 剔除异常值...")
    temp_data_cols = ['图像.最高温', '图像.最低温', '图像.平均温', 'P1.最高温', 'P2.最高温']
    temp_data_cols = [c for c in temp_data_cols if c in df_temp.columns]
    df_temp = remove_outliers(df_temp, temp_data_cols)

    # ---------- 合并 ----------
    print("[5/5] 按最近邻时间戳合并...")
    df_temp = df_temp.sort_values('timestamp').reset_index(drop=True)
    df_motor = df_motor.sort_values('timestamp').reset_index(drop=True)

    temp_merge = df_temp[['timestamp'] + temp_data_cols].copy()

    motor_original_cols = [c for c in df_motor.columns if c not in ['timestamp', '系统时间']]
    motor_merge = df_motor[['timestamp'] + motor_original_cols].copy()

    merged = pd.merge_asof(
        motor_merge,
        temp_merge,
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('1s')
    )

    # 丢弃未匹配行
    merged = merged.dropna(subset=temp_data_cols, how='all')

    # 格式化输出
    merged['系统时间'] = merged['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S.%f').str[:-3]
    output_cols = ['系统时间'] + [c for c in merged.columns if c not in ['timestamp', '系统时间']]
    output = merged[output_cols]

    # ---------- 保存 ----------
    if output_path is None:
        base1 = os.path.splitext(os.path.basename(temp_path))[0]
        base2 = os.path.splitext(os.path.basename(motor_path))[0]
        output_path = f"merged_{base1}_{base2}.csv"

    output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n完成! 合并结果: {len(output)} 行 × {len(output.columns)} 列")
    print(f"输出文件: {output_path}")
    print(f"列名: {output.columns.tolist()}")
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    temp_file = sys.argv[1]
    motor_file = sys.argv[2]
    out_file = sys.argv[3] if len(sys.argv) > 3 else None

    merge_data(temp_file, motor_file, out_file)
