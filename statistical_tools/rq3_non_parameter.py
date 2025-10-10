import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import rankdata

MAXIMIZE_SYSTEMS = {'h2'}

def is_maximize(system: str) -> bool:
    return system in MAXIMIZE_SYSTEMS

def calculate_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    return q3 - q1

def calculate_a12(group1, group2):
    n1, n2 = len(group1), len(group2)
    ranked = rankdata(np.concatenate((group1, group2)))
    rank1 = ranked[:n1]
    a12 = ((rank1.sum() - (n1 * (n1 + 1) / 2)) / (n1 * n2))
    return a12 if a12 > 0.5 else 1 - a12

def read_all_results_series(base_path, run, algo, system, workload):

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

def discover_workloads(base_path, system, prefer_algos):
    for alg in prefer_algos:
        root = os.path.join(base_path, 'run0', alg, system)
        if os.path.isdir(root):
            ws = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
            if ws:
                return ws
    return []

def collect_data_my_vs_all(systems, my_algo, other_algorithms, runs, base_path='../results'):
    os.makedirs('RQ3', exist_ok=True)

    for system in systems:
        workloads = discover_workloads(base_path, system, [my_algo] + list(other_algorithms))
        if not workloads:
            continue

        for opponent in other_algorithms:
            if opponent == my_algo:
                continue

            results_list = []

            for wl in workloads:
                my_vals, op_vals = [], []

                for run in range(runs):
                    s_my = read_all_results_series(base_path, run, my_algo, system, wl)
                    s_op = read_all_results_series(base_path, run, opponent, system, wl)
                    if s_my is not None:
                        v_my = best_of_run(s_my, system)
                        if not np.isnan(v_my):
                            my_vals.append(v_my)
                    if s_op is not None:
                        v_op = best_of_run(s_op, system)
                        if not np.isnan(v_op):
                            op_vals.append(v_op)

                if len(my_vals) > 0 and len(op_vals) > 0:

                    stat, p_value = stats.mannwhitneyu(my_vals, op_vals, alternative='two-sided')
                    a12_value = calculate_a12(my_vals, op_vals)

                    if p_value < 0.05 and a12_value > 0.56:
                        if np.mean(my_vals) < np.mean(op_vals):
                            label = 1 if not is_maximize(system) else -1
                        else:
                            label = -1 if not is_maximize(system) else 1
                    else:
                        label = 0

                    results = {
                        'workload': wl,
                        f'{my_algo}_Mean': np.mean(my_vals),
                        f'{opponent}_Mean': np.mean(op_vals),
                        'Wilcoxon_p': p_value,
                        'A12': a12_value,
                        'Label': label
                    }
                    results_list.append(results)

            results_df = pd.DataFrame(results_list)
            out_csv = f'RQ3/{system}_{my_algo}_vs_{opponent}.csv'
            results_df.to_csv(out_csv, index=False)
            print(f"Save：{out_csv}({len(results_df)} lines)")

def main():
    systems = ['batik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    my_algo = 'DLiSA-B4'
    other_algorithms = ['DLiSA']
    runs = 100
    collect_data_my_vs_all(systems, my_algo, other_algorithms, runs, base_path='../results')


if __name__ == "__main__":
    main()
