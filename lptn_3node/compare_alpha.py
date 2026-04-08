"""对比有/无铜电阻温度系数的LPTN拟合效果"""
import sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入主模块
from lptn_3node_final import (
    CONFIG, load_multiple, run_phase1, identify_parameters,
    simulate_fast, SimContext, FULL_PARAM_NAMES, NODE_CN
)

csv_files = sys.argv[1:] if len(sys.argv) > 1 else ['merged_data.csv']

print("=" * 60)
print("对比: 有温度系数 vs 无温度系数")
print("=" * 60)

data = load_multiple(csv_files, CONFIG)
print(f"  {len(data)} pts, {data['t'].max()-data['t'].min():.0f}s")

results = {}
for label, alpha in [("有α_cu=0.00393", 0.00393), ("无α_cu (α=0)", 0.0)]:
    print(f"\n{'='*60}")
    print(f"  模型: {label}")
    print(f"{'='*60}")

    cfg = dict(CONFIG)
    cfg['alpha_cu'] = alpha
    cfg['output_plot'] = f'lptn_result_alpha_{alpha:.5f}.png'

    t0 = time.time()
    p1 = run_phase1(data, cfg)
    params, T0, t, I, omega, Tamb, T_meas, ctx = identify_parameters(data, cfg, p1)

    ctx_eval = SimContext(t, I, omega, Tamb, T0, T_meas, cfg,
                          a_eff=p1['a_eff'], C1_fixed=p1['C1'])
    T_sim = simulate_fast(params, ctx_eval)
    elapsed = time.time() - t0

    rmse = [np.sqrt(np.mean((T_sim[i]-T_meas[i])**2)) for i in range(3)]
    maxe = [np.max(np.abs(T_sim[i]-T_meas[i])) for i in range(3)]

    lock_mask = np.abs(omega) < cfg.get('lock_speed_thresh', 0.5)
    rmse_lock = [np.sqrt(np.mean((T_sim[i,lock_mask]-T_meas[i,lock_mask])**2)) for i in range(3)]

    results[label] = {
        'params': params, 'T_sim': T_sim, 'T_meas': T_meas,
        't': t, 'I': I, 'omega': omega, 'Tamb': Tamb,
        'rmse': rmse, 'maxe': maxe, 'rmse_lock': rmse_lock,
        'elapsed': elapsed, 'p1': p1, 'lock_mask': lock_mask,
    }
    print(f"\n  耗时: {elapsed:.0f}s")
    for i in range(3):
        print(f"  {NODE_CN[i]}: RMSE={rmse[i]:.2f}°C, MaxErr={maxe[i]:.2f}°C, 堵转RMSE={rmse_lock[i]:.2f}°C")

# ── 对比表 ──
print(f"\n{'='*60}")
print("对比结果")
print(f"{'='*60}")
labels = list(results.keys())
print(f"{'指标':20s} | {labels[0]:18s} | {labels[1]:18s}")
print("-" * 62)
for i, name in enumerate(NODE_CN):
    print(f"  {name} RMSE   | {results[labels[0]]['rmse'][i]:16.2f}°C | {results[labels[1]]['rmse'][i]:16.2f}°C")
    print(f"  {name} MaxErr | {results[labels[0]]['maxe'][i]:16.2f}°C | {results[labels[1]]['maxe'][i]:16.2f}°C")
    print(f"  {name} 堵转   | {results[labels[0]]['rmse_lock'][i]:16.2f}°C | {results[labels[1]]['rmse_lock'][i]:16.2f}°C")
for k in ['a_eff']:
    print(f"  Phase-1 {k:7s} | {results[labels[0]]['p1'][k]:16.4f}   | {results[labels[1]]['p1'][k]:16.4f}")
for idx, name in [(8,'a'), (14,'a_lock'), (9,'b'), (10,'c')]:
    v0 = results[labels[0]]['params'][idx]
    v1 = results[labels[1]]['params'][idx]
    print(f"  {name:16s} | {v0:16.4f}   | {v1:16.4f}")

# ── 对比图 ──
fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True,
                         gridspec_kw={'height_ratios': [3, 3, 3, 2]})
colors = {labels[0]: ('#c0392b', '--'), labels[1]: ('#2980b9', ':')}
cm = ['#e74c3c', '#3498db', '#2ecc71']
titles = ['Node 1: Winding', 'Node 2: Driver (MOS)', 'Node 3: Shell']

t = results[labels[0]]['t']
T_meas = results[labels[0]]['T_meas']
lock_mask = results[labels[0]]['lock_mask']

for i in range(3):
    ax = axes[i]
    ax.plot(t, T_meas[i], color=cm[i], alpha=0.5, lw=1, label='Measured')
    for lb in labels:
        c, ls = colors[lb]
        rmse = results[lb]['rmse'][i]
        ax.plot(t, results[lb]['T_sim'][i], color=c, lw=1.5, ls=ls,
                label=f'{lb} (RMSE={rmse:.2f}°C)')
    if np.any(lock_mask):
        yl = ax.get_ylim()
        ax.fill_between(t, yl[0], yl[1], where=lock_mask, alpha=0.05, color='orange')
        ax.set_ylim(yl)
    ax.set_ylabel('Temp (°C)')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title(titles[i], fontweight='bold')
    ax.grid(True, alpha=0.3)

# 误差对比
ax = axes[3]
for lb in labels:
    c, ls = colors[lb]
    err = results[lb]['T_sim'][0] - T_meas[0]
    ax.plot(t, err, color=c, alpha=0.7, lw=0.8, ls=ls.replace(':', '-'),
            label=f'T1 err ({lb})')
ax.axhline(0, color='k', alpha=0.3, lw=0.5)
ax.set_ylabel('Error (°C)')
ax.set_xlabel('Time (s)')
ax.legend(fontsize=9, loc='upper left')
ax.set_title('Node 1 (Winding) Error Comparison', fontweight='bold')
ax.grid(True, alpha=0.3)

fig.suptitle('LPTN: With vs Without Copper Temperature Coefficient',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
outf = 'lptn_compare_alpha.png'
plt.savefig(outf, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n对比图: {outf}")
print("Done!")
