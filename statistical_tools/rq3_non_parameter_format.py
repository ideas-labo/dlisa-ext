import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

MAXIMIZE_SYSTEMS = {'h2', 'mysql', 'postgresql', 'tomcat', 'httpd'}

def is_maximize(system: str) -> bool:
    return system in MAXIMIZE_SYSTEMS

def calculate_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    return q3 - q1

def calculate_a12(group1, group2):
    n1, n2 = len(group1), len(group2)
    ranked = rankdata(np.concatenate((group1, group2)))
    rank1 = ranked[:n1]
    a12 = ((rank1.sum() - (n1 * (n1 + 1)) / 2) / (n1 * n2))
    return a12 if a12 > 0.5 else (1 - a12)

def read_all_results_series(base_path, run, algo, system, workload):
    """
    structure: ../results/run{run}/{algo}/{system}/{workload}/all_results.csv
    """
    f = os.path.join(base_path, f'run{run}', algo, system, workload, 'all_results.csv')
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    if df.shape[1] == 0:
        return None
    return df.iloc[:, -1].to_numpy(dtype=float)

def best_of_run(series: np.ndarray, system: str) -> float:
    if series.size == 0:
        return np.nan
    return float(np.max(series) if is_maximize(system) else np.min(series))

def discover_workloads_from_run0(base_path, system, algorithms):
    for alg in algorithms:
        root = os.path.join(base_path, 'run0', alg, system)
        if os.path.isdir(root):
            wls = sorted([d for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d))])
            if wls:
                return wls
    return []

def collect_data(systems, algorithms, runs, base_path='../results'):

    os.makedirs('RQ3', exist_ok=True)

    for system in systems:
        results_list = []

        workloads = discover_workloads_from_run0(base_path, system, algorithms)
        if not workloads:
            print(f"{system}: without found workloads, skipping")
            continue

        for wl in workloads:
            env_global_min, env_global_max = float('inf'), float('-inf')

            algo_data = {}
            for algo in algorithms:
                vals = []
                for run in range(runs):
                    s = read_all_results_series(base_path, run, algo, system, wl)
                    if s is None:
                        continue
                    best = best_of_run(s, system)
                    if not np.isnan(best):
                        vals.append(best)

                if len(vals) > 0:
                    env_global_min = min(env_global_min, min(vals))
                    env_global_max = max(env_global_max, max(vals))

                algo_data[algo] = vals

            baseline = algorithms[0]
            Wilcoxon_p = {}
            A12 = {}

            for i in range(1, len(algorithms)):
                candidate = algorithms[i]
                base_vals = algo_data.get(baseline, [])
                cand_vals = algo_data.get(candidate, [])
                if len(base_vals) > 0 and len(cand_vals) > 0:

                    stat, p_value = stats.mannwhitneyu(base_vals, cand_vals, alternative='two-sided')
                    a12_value = calculate_a12(base_vals, cand_vals)
                    Wilcoxon_p[candidate] = round(float(p_value), 3)
                    A12[candidate] = round(float(a12_value), 3)

            for algo in algorithms:
                vals = algo_data.get(algo, [])
                mean = float(np.mean(vals)) if len(vals) else float('nan')
                std  = float(np.std(vals))  if len(vals) else float('nan')
                iqr  = float(calculate_iqr(vals)) if len(vals) else float('nan')

                result = {
                    'workload': wl,
                    'Algorithm': algo,
                    'Mean (Std)': f'{mean:.3f} ({std:.3f})',
                }
                if algo == baseline:
                    result['$\\hat{A}_{12}$ ($p$ value)'] = None
                else:
                    p = Wilcoxon_p.get(algo, np.nan)
                    a = A12.get(algo, np.nan)
                    if np.isnan(p) or np.isnan(a):
                        result['$\\hat{A}_{12}$ ($p$ value)'] = None
                    else:
                        p_str = ' < 0.001' if p < 0.001 else f' = {p:.3f}'
                        result['$\\hat{A}_{12}$ ($p$ value)'] = f'{a:.3f} ($p${p_str})'

                results_list.append(result)

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f'RQ3/{system}.csv', index=False)
        print(f"Save: RQ3/{system}.csv ({len(results_df)} lines)")

def main():

    systems = ['batik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    algorithms = ['DLiSA-B4', 'DLiSA-BI', 'DLiSA-BII']
    runs = 100
    collect_data(systems, algorithms, runs, base_path='../results')


if __name__ == "__main__":
    main()
