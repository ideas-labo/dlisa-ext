import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from data_analysis.utility import Utility


def normalize_value(v):
    try:
        if pd.isna(v):
            return 'NaN'
        f = float(v)
        if abs(f - int(f)) < 1e-9:
            return str(int(f))
        else:
            return f"{f:.8f}".rstrip('0').rstrip('.')
    except Exception:
        return str(v)

def build_index_keymap_exact(config_bank: pd.DataFrame):
    knob_cols = list(config_bank.columns)
    if 'performance' in knob_cols:
        knob_cols.remove('performance')

    def _row_key(row):
        return tuple(normalize_value(row[c]) for c in knob_cols)

    keymap = {_row_key(config_bank.iloc[i]): i for i in range(len(config_bank))}
    return keymap, knob_cols


def match_indices_with_attrs(df: pd.DataFrame, knob_cols, keymap, attr_cols):
    idx_list = []
    attrs = {c: [] for c in attr_cols}
    for _, row in df.iterrows():
        key = tuple(normalize_value(row[c]) for c in knob_cols)
        if key in keymap:
            idx_list.append(keymap[key])
            for c in attr_cols:
                attrs[c].append(row[c])
    return idx_list, attrs


def _compute_surface(config_2d, performance, maximize: bool, grid_res=22):
    # unify direction; convention: high=good; runtime systems take negative
    z_raw = performance if maximize else -performance
    z_min, z_max = float(np.min(z_raw)), float(np.max(z_raw))
    z_norm = np.zeros_like(z_raw, dtype=float) if (z_max - z_min) < 1e-12 else (z_raw - z_min) / (z_max - z_min)

    x = config_2d[:, 0]
    y = config_2d[:, 1]
    gx, gy = np.meshgrid(
        np.linspace(x.min(), x.max(), grid_res),
        np.linspace(y.min(), y.max(), grid_res),
    )
    gz = griddata((x, y), z_norm, (gx, gy), method='linear')
    if np.isnan(gz).any():
        gz_nn = griddata((x, y), z_norm, (gx, gy), method='nearest')
        gz = np.where(np.isnan(gz), gz_nn, gz)
    return x, y, z_norm, gx, gy, gz

def _project_to_surface(x, y, z_norm, idx_array):
    idx = np.asarray(idx_array, dtype=int)
    px, py = x[idx], y[idx]
    pz = griddata((x, y), z_norm, (px, py), method='linear')
    if np.isnan(pz).any():
        pz_nn = griddata((x, y), z_norm, (px, py), method='nearest')
        pz = np.where(np.isnan(pz), pz_nn, pz)
    return px, py, pz

def _plot_landscape(
    save_path, system, workload,
    config_2d, perf_all,
    all_idx=None,              # candidate configurations -> red stars
    distilled_idx=None,        # purified configurations -> cyan circles
    weight_sizes=None,         # configuration level：size~weight
    selected_mask=None,        # configuration level：red circles
    mode='workload',           # 'workload' or 'config'
    maximize_systems=('h2', 'mysql', 'postgresql', 'tomcat', 'httpd'),
):
    maximize = (system in maximize_systems)
    x, y, z_norm, gx, gy, gz = _compute_surface(config_2d, perf_all, maximize, grid_res=22)

    fig = plt.figure(figsize=(8, 5.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(gx, gy, gz, cmap='Spectral', edgecolor='gray',
                           linewidth=0.25, alpha=0.5, vmin=0, vmax=1, zorder=0)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.72, aspect=18, pad=0.05)
    cbar.set_label('Normalized Throughput' if maximize else 'Normalized Negative Runtime', fontsize=12)

    # all historical candidates: red stars
    if all_idx:
        sx, sy, sz = _project_to_surface(x, y, z_norm, all_idx)

        # ax.scatter(sx, sy, sz, marker='*', s=36, c='red',
        #            edgecolors='k', linewidths=0.6, alpha=0.9,
        #            label='All Historical Candidates', depthshade=False, zorder=5)

    # distilled candidates (workload level)
    if distilled_idx:
        dx, dy, dz = _project_to_surface(x, y, z_norm, distilled_idx)

        if mode == 'workload':

            ax.scatter(dx, dy, dz, marker='o', s=90, facecolors='none',
                       edgecolors='black', linewidths=1.2, alpha=1.0,
                       label='Distilled Candidates', depthshade=False, zorder=9)

            # all historical candidates
            ax.scatter(sx, sy, sz, marker='*', s=36, c='red',
                       edgecolors='red', linewidths=0.6, alpha=0.9,
                       label='All Historical Candidates', depthshade=False, zorder=5)

        elif mode == 'config':
            # teal circles filled, size ~ weight
            if weight_sizes is not None and len(weight_sizes) == len(distilled_idx):
                w = np.asarray(weight_sizes, float)
                w = (w - w.min()) / (w.max() - w.min() + 1e-8)
                sizes = 40.0 + 180.0 * w  # [40, 220]
            else:
                sizes = 90

            ax.scatter(dx, dy, dz, marker='o', s=sizes, c='teal',
                       edgecolors='None', linewidths=0.6, alpha=0.95,
                       label='Distilled (size~weight)', depthshade=False, zorder=9)

            # selected ones with red circles
            if selected_mask is not None and np.any(selected_mask):
                sel = np.asarray(selected_mask, dtype=bool)
                sel_sizes = (sizes[sel] + 40) if isinstance(sizes, np.ndarray) else 140
                ax.scatter(dx[sel], dy[sel], dz[sel], marker='o', s=sel_sizes,
                           facecolors='none', edgecolors='red', linewidths=1.0,
                           alpha=1.0, label='Selected', depthshade=False, zorder=10)

    ax.set_xlabel('#D1', fontsize=12)
    ax.set_ylabel('#D2', fontsize=12)
    ax.set_zlim(0, 1)
    ax.view_init(elev=45, azim=135)
    # ax.legend(loc='upper right', fontsize=10, frameon=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # ax.view_init(elev=35, azim=-45)
    # plt.show()
    plt.savefig(save_path, bbox_inches='tight', format='pdf', pad_inches=0.15)
    plt.close()


def process_one_workload(run, system, workload, out_root):
    system_path = os.path.join('data_analysis', 'datasets', system)
    system_path_2d = os.path.join(system_path, 'workload_2d')
    config_dict, perf_dict = Utility.load_config_and_perf_data(system_path, system)

    if workload not in config_dict or workload not in perf_dict:
        print(f"[Skip] Missing dataset for {system}/{workload}")
        return

    config_bank_df = pd.DataFrame(config_dict[workload])
    perf_all = np.asarray(perf_dict[workload], dtype=float)

    fp_2d = os.path.join(system_path_2d, f"{workload}_2d.csv")
    if not os.path.exists(fp_2d):
        print(f"[Skip] 2D file not found: {fp_2d}")
        return
    df2 = pd.read_csv(fp_2d)
    if not {'MDS_Dim1', 'MDS_Dim2'}.issubset(df2.columns):
        print(f"[Skip] 2D columns missing in: {fp_2d}")
        return
    config_2d = df2[['MDS_Dim1', 'MDS_Dim2']].values

    base_res = os.path.join("results", f"run{run}", "DLiSA-B4", system, workload)
    all_csv = os.path.join(base_res, "all_candidates.csv")
    sim_csv = os.path.join(base_res, "purified_candidates.csv")
    if not (os.path.exists(all_csv) and os.path.exists(sim_csv)):
        print(f"[Skip] candidate files not found for {system}/{workload}")
        return

    all_df = pd.read_csv(all_csv)
    sim_df = pd.read_csv(sim_csv)

    # 精确匹配 -> indices
    keymap, knob_cols = build_index_keymap_exact(config_bank_df)
    for c in knob_cols:
        if c not in all_df.columns or c not in sim_df.columns:
            print(f"[WARN] knob column missing in candidates: {c}")
            return

    all_idx, _ = match_indices_with_attrs(all_df, knob_cols, keymap, attr_cols=[])
    distilled_idx, sim_attrs = match_indices_with_attrs(sim_df, knob_cols, keymap, attr_cols=['weight', 'selected'])
    weights = sim_attrs.get('weight', None)
    selected = np.array(sim_attrs.get('selected', []), dtype=bool) if 'selected' in sim_attrs else None

    out_dir = os.path.join(out_root, f"run{run}", "DLiSA-B4", system, workload)
    os.makedirs(out_dir, exist_ok=True)

    _plot_landscape(
        save_path=os.path.join(out_dir, "landscape_rank.pdf"),
        system=system, workload=workload,
        config_2d=config_2d, perf_all=perf_all,
        all_idx=all_idx, distilled_idx=distilled_idx,
        weight_sizes=None, selected_mask=None,
        mode='workload'
    )

    _plot_landscape(
        save_path=os.path.join(out_dir, "landscape_weight.pdf"),
        system=system, workload=workload,
        config_2d=config_2d, perf_all=perf_all,
        all_idx=all_idx, distilled_idx=distilled_idx,
        weight_sizes=weights, selected_mask=selected,
        mode='config'
    )

    print(f"[Saved] {out_dir}/landscape_rank.pdf")
    print(f"[Saved] {out_dir}/landscape_weight.pdf")

def main():
    runs = 10
    systems = ['batik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    systems = ['dconvert']
    out_root = "discussion/landscape_rank_weight"

    for run in range(5, 6):
        print(f"\n======================== Run {run} / {runs-1} ========================")
        for system in systems:
            order_file = f'order_files/order_{system}_{run}.json'
            if not os.path.exists(order_file):
                continue
            with open(order_file, 'r') as f:
                workloads = json.load(f)
            for workload in workloads:
                process_one_workload(run, system, workload, out_root=out_root)


if __name__ == "__main__":
    main()
