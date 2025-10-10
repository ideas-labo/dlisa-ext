
# import os
# import json
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
# from utils.config_manager import ConfigManager  # 不直接使用，但保留import无妨
# from data_analysis.utility import Utility
#
#
#
# def _infer_perf_col(df: pd.DataFrame) -> str:
#     return df.columns[-1]
#
#
# def build_index_keymap_exact(config_bank: pd.DataFrame):
#     knob_cols = list(config_bank.columns)
#     if 'performance' in knob_cols:
#         knob_cols.remove('performance')
#
#     def _row_key(row):
#         return tuple(str(row[c]) for c in knob_cols)
#
#     keymap = {_row_key(config_bank.iloc[i]): i for i in range(len(config_bank))}
#     return keymap, knob_cols
#
#
# def rows_to_indices_exact(df_rows: pd.DataFrame, knob_cols: list, keymap: dict) -> list:
#     for c in knob_cols:
#         if c not in df_rows.columns:
#             return []
#
#     keys = [tuple(str(v) for v in row)
#             for row in df_rows[knob_cols].itertuples(index=False, name=None)]
#     idx = [keymap[k] for k in keys if k in keymap]
#     return idx
#
#
# def normalize_perf(perf: np.ndarray, maximize: bool) -> np.ndarray:
#     z = perf.copy().astype(float)
#     if not maximize:
#         z = -z
#     z_min, z_max = np.min(z), np.max(z)
#     return (z - z_min) / (z_max - z_min + 1e-12)
#
#
# def plot_landscape_with_inits(save_path, system_name, workload,
#                               config_2d, performance, init_idx_by_algo,
#                               grid_res=20, snap_to_surface=True):
#
#     maximize = (system_name in ['h2', 'mysql', 'postgresql', 'tomcat', 'httpd'])
#     z_raw = performance if maximize else -performance
#
#     z_min, z_max = np.min(z_raw), np.max(z_raw)
#     if z_max - z_min < 1e-12:
#         z_norm = np.zeros_like(z_raw, dtype=float)
#     else:
#         z_norm = (z_raw - z_min) / (z_max - z_min)
#
#     x = config_2d[:, 0]
#     y = config_2d[:, 1]
#
#     grid_x, grid_y = np.meshgrid(
#         np.linspace(x.min(), x.max(), grid_res),
#         np.linspace(y.min(), y.max(), grid_res)
#     )
#     grid_z = griddata((x, y), z_norm, (grid_x, grid_y), method='linear')
#     if np.isnan(grid_z).any():
#         grid_z_nn = griddata((x, y), z_norm, (grid_x, grid_y), method='nearest')
#         grid_z = np.where(np.isnan(grid_z), grid_z_nn, grid_z)
#
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(
#         grid_x, grid_y, grid_z, cmap='Spectral',
#         edgecolor='k', linewidth=0.3, alpha=0.5, vmin=0, vmax=1
#     )
#     cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.02)
#     cbar.set_label('Normalized Throughput' if maximize else 'Normalized Runtime', fontsize=22)
#
#     markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
#     for i, (algo, idx_list) in enumerate(init_idx_by_algo.items()):
#         if not idx_list:
#             continue
#         idx_arr = np.asarray(idx_list, dtype=int)
#
#         z_pts = z_norm[idx_arr]
#
#         # if snap_to_surface:
#         #     z_lin = griddata((x, y), z_norm, (x[idx_arr], y[idx_arr]), method='linear')
#         #     if np.isnan(z_lin).any():
#         #         z_nn = griddata((x, y), z_norm, (x[idx_arr], y[idx_arr]), method='nearest')
#         #         z_lin = np.where(np.isnan(z_lin), z_nn, z_lin)
#         #     z_pts = z_lin
#
#         ax.scatter(
#             x[idx_arr], y[idx_arr], z_pts,
#             marker=markers[i % len(markers)], s=60,
#             edgecolors='k', linewidths=0.6, alpha=0.95,
#             label=f'{algo} init (n={len(idx_arr)})'
#         )
#
#     ax.set_xlabel('#D1', fontsize=22, labelpad=12)
#     ax.set_ylabel('#D2', fontsize=22, labelpad=12)
#     ax.view_init(elev=35, azim=220)
#     ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, bbox_inches='tight', format='pdf')
#     plt.close()
#
#
#
#
#
# def main():
#     runs = 10
#     # systems = ['batik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
#     algorithms = ['SEED-EA', 'DSOGA', 'FEMOSAA', 'LiDOS', 'DLiSA', 'DLiSA-B4']
#     systems = ['h2']
#
#     for run in range(5, runs):
#         print(f"\n======================== Run {run}/{runs} ==========================")
#         for system in systems:
#             system_path = os.path.join('data_analysis', 'datasets', system)
#             system_path_2d = os.path.join(system_path, 'workload_2d')
#
#             config_dict, perf_dict = Utility.load_config_and_perf_data(system_path, system)
#
#             workload_order_file = f'order_files/order_{system}_{run}.json'
#             if not os.path.exists(workload_order_file):
#                 print(f"[Skip] order file not found: {workload_order_file}")
#                 continue
#             with open(workload_order_file, 'r') as f:
#                 workloads = json.load(f)
#
#             for workload in workloads:
#                 if workload not in config_dict or workload not in perf_dict:
#                     print(f"[Skip] Missing dataset for {system}/{workload}")
#                     continue
#                 config_bank = pd.DataFrame(config_dict[workload])
#                 if config_bank.empty:
#                     print(f"[Skip] Empty config bank for {system}/{workload}")
#                     continue
#
#                 keymap, knob_cols = build_index_keymap_exact(config_bank)
#
#                 file_path_2d = os.path.join(system_path_2d, f"{workload}_2d.csv")
#                 if not os.path.exists(file_path_2d):
#                     print(f"[Skip] 2D file not found: {file_path_2d}")
#                     continue
#                 df_2d = pd.read_csv(file_path_2d)
#                 if not {'MDS_Dim1', 'MDS_Dim2'}.issubset(df_2d.columns):
#                     print(f"[Skip] 2D columns missing in: {file_path_2d}")
#                     continue
#                 config_2d = df_2d[['MDS_Dim1', 'MDS_Dim2']].values
#                 perf_all = np.asarray(perf_dict[workload])
#
#                 init_idx_by_algo = {}
#                 for algo in algorithms:
#                     all_results = os.path.join("results", f"run{run}", algo, system, workload, "all_results.csv")
#                     if not os.path.exists(all_results):
#                         continue
#
#                     df_alg = pd.read_csv(all_results)
#                     perf_col = _infer_perf_col(df_alg)
#                     cfg_cols = [c for c in df_alg.columns if c != perf_col]
#
#                     if all(c in cfg_cols for c in knob_cols):
#                         init_cfgs = df_alg.loc[:29, knob_cols]
#                         init_indices = rows_to_indices_exact(init_cfgs, knob_cols, keymap)
#                     else:
#                         missing = [c for c in knob_cols if c not in cfg_cols]
#                         print(f"[WARN] {algo}-{system}-{workload}: missing columns for exact match: {missing}")
#                         init_indices = []
#
#                     init_idx_by_algo[algo] = init_indices
#
#                     # save_dir_algo = os.path.join("landscape_analysis", f"run{run}", algo, system, workload)
#                     # os.makedirs(save_dir_algo, exist_ok=True)
#                     # (df_alg.iloc[:30]).to_csv(os.path.join(save_dir_algo, "init_30.csv"), index=False)
#
#                 if not any(len(v) > 0 for v in init_idx_by_algo.values()):
#                     print(f"[Info] No init points matched for {system}/{workload} in run{run}.")
#                     continue
#
#                 out_pdf = os.path.join("landscape_analysis", f"run{run}", "ALL", system,
#                                        f"{ workload}_inits.pdf")
#                 plot_landscape_with_inits(out_pdf, system, workload, config_2d, perf_all, init_idx_by_algo)
#                 print(f"[Saved] {out_pdf}")
#
#
# if __name__ == "__main__":
#     main()


import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.patheffects as pe

from data_analysis.utility import Utility
# from utils.config_manager import ConfigManager


def _infer_perf_col(df: pd.DataFrame) -> str:
    return df.columns[-1]

def build_index_keymap_exact(config_bank: pd.DataFrame):
    knob_cols = list(config_bank.columns)
    if 'performance' in knob_cols:
        knob_cols.remove('performance')
    def _row_key(row):
        return tuple(str(row[c]) for c in knob_cols)
    keymap = { _row_key(config_bank.iloc[i]): i for i in range(len(config_bank)) }
    return keymap, knob_cols

def rows_to_indices_exact(df_rows: pd.DataFrame, knob_cols: list, keymap: dict) -> list:
    for c in knob_cols:
        if c not in df_rows.columns:
            return []
    keys = [tuple(str(v) for v in row)
            for row in df_rows[knob_cols].itertuples(index=False, name=None)]
    return [keymap[k] for k in keys if k in keymap]

# ---------- single algorithm visualization（marker + color + label） ----------

def plot_landscape_with_inits_single(save_path, system_name, workload,
                                     config_2d, performance, init_indices,
                                     marker='o', color='tab:blue', label=None,
                                     grid_res=20, snap_to_surface=True, title=None):

    plt.rcParams.update({
        # "axes.labelsize": 28,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    # unify direction; convention: high=good; runtime systems take negative
    maximize = (system_name in ['h2', 'mysql', 'postgresql', 'tomcat', 'httpd'])
    z_raw = performance if maximize else -performance
    z_min, z_max = np.min(z_raw), np.max(z_raw)
    z_norm = np.zeros_like(z_raw, dtype=float) if (z_max - z_min < 1e-12) else (z_raw - z_min) / (z_max - z_min)

    x = config_2d[:, 0]
    y = config_2d[:, 1]

    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), grid_res),
        np.linspace(y.min(), y.max(), grid_res)
    )
    grid_z = griddata((x, y), z_norm, (grid_x, grid_y), method='linear')
    if np.isnan(grid_z).any():
        grid_z_nn = griddata((x, y), z_norm, (grid_x, grid_y), method='nearest')
        grid_z = np.where(np.isnan(grid_z), grid_z_nn, grid_z)

    fig = plt.figure(figsize=(10, 7))
    # if title:
    #     fig.suptitle(title, fontsize=16, y=0.98)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='Spectral',
                           edgecolor='gray', linewidth=0.03, alpha=0.5, vmin=0, vmax=1, zorder=0)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.02)
    cbar.set_label('Normalized Throughput' if maximize else 'Normalized Negative Runtime', fontsize=22)

    # initial points（marker+color+label）
    if init_indices:
        idx_arr = np.asarray(init_indices, dtype=int)
        z_pts = z_norm[idx_arr]
        if snap_to_surface:
            z_lin = griddata((x, y), z_norm, (x[idx_arr], y[idx_arr]), method='linear')
            if np.isnan(z_lin).any():
                z_nn = griddata((x, y), z_norm, (x[idx_arr], y[idx_arr]), method='nearest')
                z_lin = np.where(np.isnan(z_lin), z_nn, z_lin)
            z_pts = z_lin

        sc = ax.scatter(x[idx_arr], y[idx_arr], z_pts,
                   marker=marker, s=120,
                   facecolors=color, edgecolors='k', linewidths=1.2, alpha=1,
                   label=label, depthshade=False, zorder=999)
        if hasattr(sc, "set_sort_zpos"):
            sc.set_sort_zpos(999999999)
        ax.legend(loc='upper left', bbox_to_anchor=(0.95, 0.95), borderaxespad=0., fontsize=20)

    ax.set_xlabel('#D1', fontsize=22, labelpad=12)
    ax.set_ylabel('#D2', fontsize=22, labelpad=12)
    # plt.show()
    ax.view_init(elev=35, azim=220)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.close()


def main():
    runs = 10
    # systems = ['batik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    systems = ['h2']
    algorithms = ['SEED-EA', 'DSOGA', 'FEMOSAA', 'LiDOS', 'DLiSA', 'DLiSA-B4']

    marker_cycle = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
    color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                    'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    color_cycle = ['#111111', '#ff4b00', '#00b3ff', '#00c853',
                   '#7e57c2', '#d81b60', '#795548', '#ffb300']
    marker_map = {algo: marker_cycle[i % len(marker_cycle)] for i, algo in enumerate(algorithms)}
    color_map = {algo: color_cycle[i % len(color_cycle)]   for i, algo in enumerate(algorithms)}

    print("Style mapping (algo -> marker, color):")
    for a in algorithms:
        print(f"  {a:10s} -> {marker_map[a]}, {color_map[a]}")

    for run in range(5, 6):
        print(f"\n======================== Run {run}/{runs} ==========================")
        for system in systems:
            system_path = os.path.join('data_analysis', 'datasets', system)
            system_path_2d = os.path.join(system_path, 'workload_2d')

            config_dict, perf_dict = Utility.load_config_and_perf_data(system_path, system)

            workload_order_file = f'order_files/order_{system}_{run}.json'
            if not os.path.exists(workload_order_file):
                print(f"[Skip] {workload_order_file} not found.")
                continue
            with open(workload_order_file, 'r') as f:
                workloads = json.load(f)

            for workload in workloads:
                if workload not in config_dict or workload not in perf_dict:
                    print(f"[Skip] Missing dataset for {system}/{workload}")
                    continue

                config_bank = pd.DataFrame(config_dict[workload])
                if config_bank.empty:
                    print(f"[Skip] Empty config bank for {system}/{workload}")
                    continue
                keymap, knob_cols = build_index_keymap_exact(config_bank)

                file_path_2d = os.path.join(system_path_2d, f"{workload}_2d.csv")
                if not os.path.exists(file_path_2d):
                    print(f"[Skip] 2D file not found: {file_path_2d}")
                    continue
                df2 = pd.read_csv(file_path_2d)
                if not {'MDS_Dim1', 'MDS_Dim2'}.issubset(df2.columns):
                    print(f"[Skip] 2D columns missing in: {file_path_2d}")
                    continue
                config_2d = df2[['MDS_Dim1', 'MDS_Dim2']].values
                perf_all = np.asarray(perf_dict[workload])

                for algo in algorithms:
                    base_path = os.path.join("results", f"run{run}", algo, system, workload)
                    all_results = os.path.join(base_path, "all_results.csv")
                    if not os.path.exists(all_results):
                        continue

                    df_alg = pd.read_csv(all_results)
                    perf_col = _infer_perf_col(df_alg)
                    cfg_cols = [c for c in df_alg.columns if c != perf_col]

                    if all(c in cfg_cols for c in knob_cols):
                        init_cfgs = df_alg.loc[:29, knob_cols]
                        init_indices = rows_to_indices_exact(init_cfgs, knob_cols, keymap)
                    else:
                        missing = [c for c in knob_cols if c not in cfg_cols]
                        print(f"[WARN] {algo}-{system}-{workload}: missing columns for exact match: {missing}")
                        init_indices = []

                    os.makedirs(base_path, exist_ok=True)
                    (df_alg.iloc[:30]).to_csv(os.path.join(base_path, "init_30.csv"), index=False)

                    label_text = f"{algo} init (n={len(init_indices)})"
                    if algo == 'DLiSA-B4':
                        label_text = "DLiSA"
                    elif algo == 'DLiSA':
                        label_text = r"$\mathrm{DLiSA}_{\mathit{ICSE}}$"
                    elif algo == 'SEED-EA':
                        label_text = 'Seed-EA'
                    elif algo == 'DSOGA':
                        label_text = 'D-SOGA'
                    elif algo == 'FEMOSAA':
                        label_text = 'FEMOSAA'
                    elif algo == 'LiDOS':
                        label_text = 'LiDOS'

                    out_pdf = os.path.join("discussion/landscape_seeds", f"run{run}", algo, system,
                                           f"{workload}-{label_text}.pdf")
                    plot_landscape_with_inits_single(
                        save_path=out_pdf,
                        system_name=system,
                        workload=workload,
                        config_2d=config_2d,
                        performance=perf_all,
                        init_indices=init_indices,
                        marker=marker_map[algo],
                        color=color_map[algo],
                        label=label_text,
                        title=f"{system}/{workload} — {algo}"
                    )
                    print(f"[Saved] {out_pdf}")


if __name__ == "__main__":
    main()
