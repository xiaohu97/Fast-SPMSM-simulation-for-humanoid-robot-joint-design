"""
三节点 LPTN / RC 热网络参数辨识工具
====================================
适用于: 机器人关节电机 (PMSM) + 一体化驱动板

节点定义:
  节点1: 电机线圈/铁芯  (T1)  ← Ch1_Temp_C   (绕组温度传感器)
  节点2: 驱动板         (T2)  ← Ch1_MOS_C    (MOS管温度传感器)
  节点3: 外壳           (T3)  ← P1.最高温     (红外热像仪P1区域)
  边界:  环境温度       (Tamb) ← 图像.最低温   (红外最低温 ≈ 环境)

热源模型:
  P1 = a·I² + b·|ω| + c·ω²    (铜耗 + 铁耗/风摩)
  P2 = d·I² + e·|I| + f        (导通损耗 + 开关损耗 + 待机)
  P3 ≈ 0                        (外壳无主动热源)

状态方程 (3节点RC网络):
  C1·dT1/dt = P1 - (T1-T2)/R12 - (T1-T3)/R13
  C2·dT2/dt = P2 - (T2-T1)/R12 - (T2-T3)/R23 - (T2-Tamb)/R2a
  C3·dT3/dt =    + (T1-T3)/R13 + (T2-T3)/R23 - (T3-Tamb)/R3a

待辨识参数 (14个):
  热容:  C1, C2, C3           [J/K]
  热阻:  R12, R13, R23, R2a, R3a  [K/W]
  损耗:  a, b, c, d, e, f     [各自单位见下方]

用法:
  python lptn_3node_final.py <合并CSV文件1> [合并CSV文件2] ...

  若提供多个文件，会按时间顺序拼接后统一辨识。
  输入CSV需要包含以下列:
    系统时间, Ch1_Temp_C, Ch1_MOS_C, Ch1_Cur_A, Ch1_Spd_rads,
    P1.最高温, 图像.最低温

示例:
  python lptn_3node_final.py merged_data_2.csv merged_data-2.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置区 —— 根据实际情况调整
# ============================================================
CONFIG = {
    # 降采样步长 (秒)。越小越精确但辨识越慢。
    # 建议: 首次调试用 5.0, 最终精调用 1.0~2.0
    'resample_dt': 5.0,

    # 列名映射: 左边是模型变量名, 右边是CSV列名
    # 如果你的传感器布局不同, 只需修改这里
    'col_map': {
        'T1':   'Ch1_Temp_C',      # 节点1温度 (绕组)
        'T2':   'Ch1_MOS_C',       # 节点2温度 (驱动板MOS)
        'T3':   'P1.最高温',        # 节点3温度 (外壳红外P1)
        'Tamb': '图像.最低温',       # 环境温度
        'I':    'Ch1_Cur_A',       # 相电流 (A)
        'omega':'Ch1_Spd_rads',    # 角速度 (rad/s)
    },

    # 时间列名
    'time_col': '系统时间',
    'time_format': '%Y/%m/%d %H:%M:%S.%f',

    # 参数边界 [下限, 上限]
    'bounds': {
        'C1':  (1.0,   300.0),     # 线圈热容 (J/K)
        'C2':  (0.5,   200.0),     # 驱动板热容 (J/K)
        'C3':  (1.0,   800.0),     # 外壳热容 (J/K)
        'R12': (0.05,  30.0),      # 线圈↔驱动板热阻 (K/W)
        'R13': (0.05,  30.0),      # 线圈↔外壳热阻 (K/W)
        'R23': (0.1,   50.0),      # 驱动板↔外壳热阻 (K/W)
        'R2a': (0.1,   50.0),      # 驱动板↔环境热阻 (K/W)
        'R3a': (0.1,   50.0),      # 外壳↔环境热阻 (K/W)
        'a':   (0.001, 10.0),      # I² 铜耗系数 (W/A²)
        'b':   (0.0,   2.0),       # |ω| 铁耗线性项 (W·s/rad)
        'c':   (0.0,   0.5),       # ω² 铁耗二次项 (W·s²/rad²)
        'd':   (0.0,   5.0),       # I² 驱动板导通损耗 (W/A²)
        'e':   (0.0,   5.0),       # |I| 驱动板开关损耗 (W/A)
        'f':   (0.0,   15.0),      # 驱动板待机损耗 (W)
    },

    # 差分进化优化设置
    'de_seeds': [42, 123, 7],      # 多种子并行 (取最优)
    'de_maxiter': 200,             # 每个种子的最大迭代
    'de_popsize': 15,              # 种群大小

    # 输出路径
    'output_plot': 'lptn_result.png',
    'output_params': 'lptn_params.csv',
}


# ============================================================
# 1. 数据加载与预处理
# ============================================================
def load_and_preprocess(csv_path, config):
    """加载合并后的CSV，提取热模型所需列，降采样"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['timestamp'] = pd.to_datetime(
        df[config['time_col']], format=config['time_format']
    )
    t0 = df['timestamp'].iloc[0]
    df['t_sec'] = (df['timestamp'] - t0).dt.total_seconds()

    cm = config['col_map']
    data = pd.DataFrame({
        't':    df['t_sec'],
        'T1':   df[cm['T1']].astype(float),
        'T2':   df[cm['T2']].astype(float),
        'T3':   df[cm['T3']].astype(float),
        'Tamb': df[cm['Tamb']].astype(float),
        'I':    df[cm['I']].astype(float),
        'omega':df[cm['omega']].astype(float),
    })

    dt = config['resample_dt']
    data['t_bin'] = (data['t'] / dt).astype(int) * dt
    data_rs = data.groupby('t_bin').mean().reset_index()
    data_rs['t'] = data_rs['t_bin']
    data_rs.drop(columns=['t_bin'], inplace=True)
    return data_rs


def load_multiple(csv_paths, config):
    """加载并拼接多个数据集"""
    dfs = []
    for path in csv_paths:
        d = load_and_preprocess(path, config)
        if dfs:
            gap = 5.0
            d['t'] = d['t'] + dfs[-1]['t'].max() + gap
        dfs.append(d)
    combined = pd.concat(dfs, ignore_index=True)
    return combined


# ============================================================
# 2. 正向仿真 (Forward Euler)
# ============================================================
def simulate(params, t_data, I_data, omega_data, Tamb_data, T0):
    """Forward Euler 正向仿真三节点温度"""
    C1, C2, C3, R12, R13, R23, R2a, R3a, a, b, c, d, e, f = params
    N = len(t_data)
    T_sim = np.zeros((3, N))
    T_sim[:, 0] = T0

    for k in range(N - 1):
        dt = t_data[k+1] - t_data[k]
        if dt <= 0 or dt > 30:
            T_sim[:, k+1] = T_sim[:, k]
            continue

        T1, T2, T3 = T_sim[0, k], T_sim[1, k], T_sim[2, k]
        Ik, wk, Ta = I_data[k], omega_data[k], Tamb_data[k]

        P1 = a * Ik**2 + b * abs(wk) + c * wk**2
        P2 = d * Ik**2 + e * abs(Ik) + f

        dT1 = (P1 - (T1-T2)/R12 - (T1-T3)/R13) / C1
        dT2 = (P2 - (T2-T1)/R12 - (T2-T3)/R23 - (T2-Ta)/R2a) / C2
        dT3 = ((T1-T3)/R13 + (T2-T3)/R23 - (T3-Ta)/R3a) / C3

        T_sim[0, k+1] = T1 + dT1 * dt
        T_sim[1, k+1] = T2 + dT2 * dt
        T_sim[2, k+1] = T3 + dT3 * dt

    return t_data, T_sim


# ============================================================
# 3. 参数辨识
# ============================================================
def cost_function(params_vec, t_data, I_data, omega_data, Tamb_data, T_meas, T0, weights):
    """加权 MSE 目标函数"""
    try:
        _, T_sim = simulate(params_vec, t_data, I_data, omega_data, Tamb_data, T0)
        if np.any(np.isnan(T_sim)):
            return 1e6
        err = 0.0
        for i in range(3):
            err += weights[i] * np.mean((T_sim[i] - T_meas[i])**2)
        return err
    except Exception:
        return 1e6


def identify_parameters(data, config):
    """差分进化 + L-BFGS-B 多种子辨识"""
    t = data['t'].values
    I = data['I'].values
    omega = data['omega'].values
    Tamb = data['Tamb'].values
    T_meas = np.array([data['T1'].values, data['T2'].values, data['T3'].values])
    T0 = [T_meas[0, 0], T_meas[1, 0], T_meas[2, 0]]

    # 归一化权重
    ranges = [max(T_meas[i].max() - T_meas[i].min(), 1.0) for i in range(3)]
    weights = [1.0 / r**2 for r in ranges]

    print(f"  T1 range: {ranges[0]:.1f}°C, T2 range: {ranges[1]:.1f}°C, T3 range: {ranges[2]:.1f}°C")

    bounds_dict = config['bounds']
    bounds_list = [
        bounds_dict['C1'], bounds_dict['C2'], bounds_dict['C3'],
        bounds_dict['R12'], bounds_dict['R13'], bounds_dict['R23'],
        bounds_dict['R2a'], bounds_dict['R3a'],
        bounds_dict['a'], bounds_dict['b'], bounds_dict['c'],
        bounds_dict['d'], bounds_dict['e'], bounds_dict['f'],
    ]

    best_cost = 1e10
    best_x = None

    for seed_val in config['de_seeds']:
        print(f"\n  DE (seed={seed_val})...", end=' ')
        result_de = differential_evolution(
            cost_function, bounds_list,
            args=(t, I, omega, Tamb, T_meas, T0, weights),
            seed=seed_val,
            maxiter=config['de_maxiter'],
            tol=1e-8,
            polish=False,
            workers=1,
            disp=False,
            popsize=config['de_popsize'],
            mutation=(0.5, 1.5),
            recombination=0.9,
        )
        print(f"cost={result_de.fun:.6f}", end=' → ')

        result_lb = minimize(
            cost_function, result_de.x,
            args=(t, I, omega, Tamb, T_meas, T0, weights),
            method='L-BFGS-B',
            bounds=bounds_list,
            options={'maxiter': 500, 'ftol': 1e-12},
        )
        print(f"refined={result_lb.fun:.6f}")

        if result_lb.fun < best_cost:
            best_cost = result_lb.fun
            best_x = result_lb.x

    print(f"\n  Best cost: {best_cost:.6f}")
    return best_x, T0, t, I, omega, Tamb, T_meas


# ============================================================
# 4. 结果输出
# ============================================================
PARAM_NAMES = ['C1', 'C2', 'C3', 'R12', 'R13', 'R23', 'R2a', 'R3a',
               'a', 'b', 'c', 'd', 'e', 'f']
PARAM_UNITS = ['J/K','J/K','J/K','K/W','K/W','K/W','K/W','K/W',
               'W/A²','W·s/rad','W·s²/rad²','W/A²','W/A','W']
PARAM_DESC = [
    'Winding thermal capacitance',
    'Driver thermal capacitance',
    'Shell thermal capacitance',
    'Winding↔Driver thermal resistance',
    'Winding↔Shell thermal resistance',
    'Driver↔Shell thermal resistance',
    'Driver↔Ambient thermal resistance',
    'Shell↔Ambient thermal resistance',
    'Copper loss coeff (I²)',
    'Iron loss coeff (|ω|)',
    'Iron loss coeff (ω²)',
    'Driver conduction loss (I²)',
    'Driver switching loss (|I|)',
    'Driver standby loss (const)',
]
NODE_NAMES = ['Node1 (Winding)', 'Node2 (Driver)', 'Node3 (Shell)']
NODE_NAMES_CN = ['节点1 (绕组)', '节点2 (驱动板)', '节点3 (外壳)']


def print_results(params):
    """打印辨识结果和模型方程"""
    C1, C2, C3, R12, R13, R23, R2a, R3a, a, b, c, d, e, f = params

    print("\n" + "=" * 60)
    print("IDENTIFIED PARAMETERS")
    print("=" * 60)
    for name, val, unit in zip(PARAM_NAMES, params, PARAM_UNITS):
        print(f"  {name:6s} = {val:12.6f}  {unit}")

    print("\n" + "=" * 60)
    print("MODEL EQUATIONS")
    print("=" * 60)
    print(f"""
  P1 = {a:.4f}·I² + {b:.4f}·|ω| + {c:.6f}·ω²   [W]
  P2 = {d:.4f}·I² + {e:.4f}·|I| + {f:.4f}          [W]

  {C1:.2f}·dT1/dt = P1 - (T1-T2)/{R12:.4f} - (T1-T3)/{R13:.4f}
  {C2:.2f}·dT2/dt = P2 - (T2-T1)/{R12:.4f} - (T2-T3)/{R23:.4f} - (T2-Tamb)/{R2a:.4f}
  {C3:.2f}·dT3/dt =    + (T1-T3)/{R13:.4f} + (T2-T3)/{R23:.4f} - (T3-Tamb)/{R3a:.4f}

  Time constants:
    τ1 ≈ {C1 * (1/(1/R12 + 1/R13)):.1f} s   (winding)
    τ2 ≈ {C2 * (1/(1/R12 + 1/R23 + 1/R2a)):.1f} s   (driver)
    τ3 ≈ {C3 * (1/(1/R13 + 1/R23 + 1/R3a)):.1f} s   (shell)
""")


def evaluate_and_plot(params, T0, t, I, omega, Tamb, T_meas, config):
    """仿真对比 + 绘图 + 保存参数"""
    _, T_sim = simulate(params, t, I, omega, Tamb, T0)

    print("FITTING ACCURACY")
    print("-" * 40)
    for i in range(3):
        rmse = np.sqrt(np.mean((T_sim[i] - T_meas[i])**2))
        mae = np.mean(np.abs(T_sim[i] - T_meas[i]))
        maxe = np.max(np.abs(T_sim[i] - T_meas[i]))
        print(f"  {NODE_NAMES_CN[i]}: RMSE={rmse:.2f}°C, MAE={mae:.2f}°C, MaxErr={maxe:.2f}°C")

    # 损耗
    P1 = params[8]*I**2 + params[9]*np.abs(omega) + params[10]*omega**2
    P2 = params[11]*I**2 + params[12]*np.abs(I) + params[13]

    # ---- 绘图 ----
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    colors_m = ['#e74c3c', '#3498db', '#2ecc71']
    colors_s = ['#c0392b', '#2980b9', '#27ae60']

    titles = ['Node 1: Winding / Iron Core',
              'Node 2: Driver Board (MOS)',
              'Node 3: Shell / Housing']
    labels_m = ['T1 measured (winding)', 'T2 measured (driver MOS)', 'T3 measured (shell IR)']

    for i in range(3):
        ax = axes[i]
        rmse = np.sqrt(np.mean((T_sim[i] - T_meas[i])**2))
        ax.plot(t, T_meas[i], color=colors_m[i], alpha=0.5, lw=0.8, label=labels_m[i])
        ax.plot(t, T_sim[i], color=colors_s[i], lw=1.5, ls='--',
                label=f'Simulated (RMSE={rmse:.2f}°C)')
        if i == 2:
            ax.plot(t, Tamb, color='gray', alpha=0.4, lw=0.8, label='Tamb')
        ax.set_ylabel('Temperature (°C)', fontsize=11)
        ax.legend(fontsize=10, loc='upper left')
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax2 = ax.twinx()
    l1 = ax.plot(t, I, color='#9b59b6', alpha=0.3, lw=0.5, label='Current (A)')
    l2 = ax2.plot(t, omega, color='#e67e22', alpha=0.3, lw=0.5, label='Speed (rad/s)')
    l3 = ax.plot(t, P1, color='#e74c3c', alpha=0.6, lw=1.0, label='P1 loss (W)')
    l4 = ax.plot(t, P2, color='#3498db', alpha=0.6, lw=1.0, label='P2 loss (W)')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Current (A) / Power Loss (W)', fontsize=11)
    ax2.set_ylabel('Angular Speed (rad/s)', fontsize=11)
    lines = l1 + l2 + l3 + l4
    ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='upper left')
    ax.set_title('Inputs: Current, Speed & Estimated Losses', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.suptitle('3-Node LPTN Thermal Network: Identification vs Measurement',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(config['output_plot'], dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {config['output_plot']}")

    # 保存参数 CSV
    pd.DataFrame({
        'Parameter': PARAM_NAMES,
        'Value': params,
        'Unit': PARAM_UNITS,
        'Description': PARAM_DESC,
    }).to_csv(config['output_params'], index=False, encoding='utf-8-sig')
    print(f"Parameters saved: {config['output_params']}")


# ============================================================
# 5. 主入口
# ============================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_files = sys.argv[1:]
    print("=" * 60)
    print("3-Node LPTN Thermal Network Parameter Identification")
    print("=" * 60)

    print(f"\n[1/3] Loading {len(csv_files)} file(s)...")
    data = load_multiple(csv_files, CONFIG)
    print(f"  Samples: {len(data)}, Time: {data['t'].min():.0f}~{data['t'].max():.0f} s")
    print(f"  T1: {data['T1'].min():.1f}~{data['T1'].max():.1f}°C")
    print(f"  T2: {data['T2'].min():.1f}~{data['T2'].max():.1f}°C")
    print(f"  T3: {data['T3'].min():.1f}~{data['T3'].max():.1f}°C")
    print(f"  I:  {data['I'].min():.2f}~{data['I'].max():.2f} A")
    print(f"  ω:  {data['omega'].min():.2f}~{data['omega'].max():.2f} rad/s")

    print(f"\n[2/3] Parameter identification...")
    params, T0, t, I, omega, Tamb, T_meas = identify_parameters(data, CONFIG)
    print_results(params)

    print("[3/3] Validation & visualization...")
    evaluate_and_plot(params, T0, t, I, omega, Tamb, T_meas, CONFIG)

    print("\nDone!")
