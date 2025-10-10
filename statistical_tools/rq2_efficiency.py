import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


MAXIMIZE_SYSTEMS = {'h2', 'mysql', 'postgresql', 'tomcat', 'httpd'}

def is_maximize(system: str) -> bool:
    return system in MAXIMIZE_SYSTEMS

def ensure_convergence(series: np.ndarray, system: str) -> np.ndarray:

    series = series.copy()
    if is_maximize(system):
        for i in range(1, len(series)):
            if series[i] < series[i - 1]:
                series[i] = series[i - 1]
    else:
        for i in range(1, len(series)):
            if series[i] > series[i - 1]:
                series[i] = series[i - 1]
    return series

def first_hit_index(perf: np.ndarray, target: float, system: str) -> Optional[int]:

    if is_maximize(system):
        idx = np.where(perf >= target)[0]
    else:
        idx = np.where(perf <= target)[0]
    return int(idx[0] + 1) if idx.size > 0 else None

def read_perf_series(base_path: str, run: int, algo: str, system: str, workload: str,
                     consider_budget: Optional[int] = None) -> Optional[np.ndarray]:

    f = os.path.join(base_path, f'run{run}', algo, system, workload, 'all_results.csv')
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    if df.shape[1] == 0:
        return None
    s = df.iloc[:, -1].to_numpy(dtype=float)
    if consider_budget is not None:
        s = s[:consider_budget]
    return s

def mean_curve_across_runs(series_list: List[np.ndarray]) -> Optional[np.ndarray]:

    if not series_list:
        return None
    min_len = min(len(s) for s in series_list)
    if min_len == 0:
        return None
    stack = np.vstack([s[:min_len] for s in series_list])
    return stack.mean(axis=0)

def discover_workloads(base_path: str, systems: List[str], preferred_algo: str,
                       fallback_algos: List[str]) -> Dict[str, List[str]]:

    result = {}
    for sys in systems:
        candidates = [preferred_algo] + list(fallback_algos)
        workloads = []
        for alg in candidates:
            root = os.path.join(base_path, 'run0', alg, sys)
            if os.path.isdir(root):
                ws = sorted([d for d in os.listdir(root)
                             if os.path.isdir(os.path.join(root, d))])
                if ws:
                    workloads = ws
                    break
        result[sys] = workloads
    return result

def pairwise_efficiency_using_all_results(
    systems: List[str],
    my_algo: str,
    other_algorithms: List[str],
    runs: int,
    base_path: str = '../results',
    out_dir: str = 'RQ2',
    consider_budget: Optional[int] = None,
) -> None:

    os.makedirs(out_dir, exist_ok=True)


    workloads_map = discover_workloads(base_path, systems, preferred_algo=my_algo, fallback_algos=other_algorithms)

    for system in systems:
        workloads = workloads_map.get(system, [])
        if not workloads:
            print(f"{system}: without found workload")
            continue

        for opponent in other_algorithms:
            if opponent == my_algo:
                continue

            rows = []
            for wl in workloads:
                opp_runs = []
                mine_runs = []

                for r in range(runs):
                    s_opp = read_perf_series(base_path, r, opponent, system, wl, consider_budget)
                    s_mine = read_perf_series(base_path, r, my_algo, system, wl, consider_budget)
                    if s_opp is not None and len(s_opp) > 0:
                        opp_runs.append(ensure_convergence(s_opp, system))
                    if s_mine is not None and len(s_mine) > 0:
                        mine_runs.append(ensure_convergence(s_mine, system))

                if not opp_runs or not mine_runs:
                    print(f"{system}/{wl}: lack of data（{opponent} runs={len(opp_runs)}, {my_algo} runs={len(mine_runs)}），skipping workload")
                    continue

                opp_mean = mean_curve_across_runs(opp_runs)
                mine_mean = mean_curve_across_runs(mine_runs)
                if opp_mean is None or mine_mean is None:
                    print(f"{system}/{wl}: empty mean curve, skipping workload")
                    continue

                L = min(len(opp_mean), len(mine_mean))
                opp_mean = opp_mean[:L]
                mine_mean = mine_mean[:L]

                opp_conv = ensure_convergence(opp_mean, system)
                mine_conv = ensure_convergence(mine_mean, system)

                T = float(opp_conv[-1])
                b = first_hit_index(opp_conv, T, system)

                m = first_hit_index(mine_conv, T, system)

                if b is None:
                    s = -1
                elif m is None:
                    s = -1
                else:
                    s = b / m

                if s > 1:
                    label = 1
                elif s == 1:
                    label = 0
                elif 0 < s < 1:
                    label = 999
                else:  # s == -1
                    label = -1

                rows.append({
                    'workload'   : wl,
                    'target_algo': opponent,
                    'target_b'   : b,
                    'target_T'   : T,
                    'my_algo'    : my_algo,
                    'my_m'       : m,
                    's'          : s,
                    'label'      : label
                })

            out_csv = os.path.join(out_dir, f'convergence_{system}_{my_algo}_vs_{opponent}.csv')
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"Save：{out_csv}（{len(rows)} lines）")

def main():
    systems = ['batik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    my_algo = 'DLiSA-B4'
    other_algorithms = ['DLiSA']  # ['FEMOSAA', 'SEED-EA', 'DSOGA', 'LiDOS']
    runs = 100
    base_path = '../results'
    out_dir = 'RQ2'
    consider_budget = None

    pairwise_efficiency_using_all_results(
        systems=systems,
        my_algo=my_algo,
        other_algorithms=other_algorithms,
        runs=runs,
        base_path=base_path,
        out_dir=out_dir,
        consider_budget=consider_budget
    )


if __name__ == '__main__':
    main()
