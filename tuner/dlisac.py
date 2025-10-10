import json
import os
import random
from itertools import combinations
import pandas as pd
import numpy as np

from utils.evolutionary_planning import EvolutionaryPlanner
from utils.config_manager import ConfigManager
from utils.logger import Logger
from collections import defaultdict, Counter


class DLiSACTuner:
    def __init__(self, system, workloads, run_id, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback=None):

        self.logger = None
        self.config_manager = None
        self.system = system
        self.workloads = workloads
        self.run_id = run_id
        self.optimization_goal = optimization_goal
        self.initial_seeds = initial_seeds
        self.common_seeds = common_seeds
        self.fallback = fallback

        self.max_generation = max_generation
        self.pop_size = pop_size
        self.history_optimized_population = {}
        self.common_seeds_performance = {}  # workload_name -> Dict[config_tuple -> perf]

        self.similarity_log_path = f"results/run{self.run_id}/DLiSA-C/{self.system}/similarity.json"
        self.similarity_records = {}

    def run(self):
        self.config_manager = ConfigManager(system=self.system, optimization_goal=self.optimization_goal)

        for workload_id, workload_name in enumerate(self.workloads):
            print(f"\n[DLiSA-C | RUN {self.run_id}] >>> Workload {workload_id + 1}/{len(self.workloads)}: {workload_name}")
            self.config_manager.set_workload(workload_name)

            # evaluate common_seeds
            evaluated_common_seeds = self.config_manager.evaluate(self.common_seeds)
            self.common_seeds_performance[workload_name] = {
                tuple(cfg): perf for cfg, perf in evaluated_common_seeds
            }

            header = self.config_manager.get_config_header()
            self.logger = Logger(
                base_dir=f"results/run{self.run_id}",
                algorithm="DLiSA-C",
                system=f"{self.system}",
                workload=f"{workload_name}",
                header=header
            )

            if workload_id == 0:
                if self.initial_seeds:
                    init_population = self.initial_seeds
                else:
                    init_population = self.config_manager.random_initialize_population(self.pop_size)
            else:
                init_population = self.generate_seeds_by_dlisab(workload_name)

            planner = EvolutionaryPlanner(
                config_manager=self.config_manager,
                init_population=init_population,
                max_generation=self.max_generation,
                mutation_rate=0.1,
                crossover_rate=0.9,
                common_seeds=self.common_seeds,
                pop_size=self.pop_size,
                logger=self.logger,
                fallback=self.fallback,
                evaluated_common_seeds=evaluated_common_seeds
            )
            optimized_pop = planner.evolutionary_search_config()
            self.history_optimized_population[workload_name] = optimized_pop

            if workload_id == 0:
                # 用第一个 workload 的 top-k 个体作为新的 common_seeds
                reverse = self.optimization_goal == "maximum"
                sorted_pop = sorted(optimized_pop, key=lambda x: x[1], reverse=reverse)
                top_common = sorted_pop[:self.pop_size // 2]
                self.common_seeds = [list(cfg) for cfg, _ in top_common]

                # 同时更新 common_seeds_performance（只包含第一个 workload）
                wl_name = workload_name  # 当前 workload 就是第一个 workload
                self.common_seeds_performance[wl_name] = {
                    tuple(cfg): perf for cfg, perf in top_common
                }

    def generate_seeds_by_dlisab(self, current_workload):
        """
        Generate seeds for the current workload using DLiSA.
        This method should implement the logic to generate seeds based on the history of previous workloads.
        """
        if len(self.history_optimized_population) == 1:
            only_workload = next(iter(self.history_optimized_population))
            prev_pop = self.history_optimized_population[only_workload]

            reverse = self.optimization_goal == "maximum"
            sorted_pop = sorted(prev_pop, key=lambda x: x[1], reverse=reverse)
            half = self.pop_size // 2
            selected = [list(cfg) for cfg, _ in sorted_pop[:half]]

            # 生成剩下的一半，避免重复
            existing_set = {tuple(cfg) for cfg in selected}
            filler = self.config_manager.random_initialize_population(self.pop_size - len(selected),
                                                                      existing_configs=existing_set)
            init_population = selected + filler
        else:
            similar_workloads = self.rank_workload_similarity_analysis(current_workload, beta=0.3)
            # self.similarity_records[current_workload] = similar_workloads
            # self.save_similarity_records()

            if not similar_workloads:
                init_population = self.config_manager.random_initialize_population(self.pop_size)
            else:
                init_population = self.weighted_config_seeding(similar_workloads, current_workload)

        return init_population



    def rank_workload_similarity_analysis(self, current_workload, beta):

        if current_workload not in self.common_seeds_performance:
            raise ValueError(f"Current workload {current_workload} common_seeds data not available.")

        current_perf_dict = self.common_seeds_performance[current_workload]
        current_keys = set(current_perf_dict.keys())

        similar_workloads = {}
        self.similarity_records[current_workload] = {}  # 初始化当前 workload 的记录

        for his_wl, his_perf_dict in self.common_seeds_performance.items():
            if his_wl == current_workload:
                continue

            common = set(his_perf_dict.keys()) & current_keys

            sim = self.calculate_similarity(his_perf_dict, current_perf_dict, common, beta)


            self.similarity_records.setdefault(current_workload, {})[his_wl] = sim


            if sim >= beta:
                similar_workloads[his_wl] = sim

        self.save_similarity_records()  # ✅ 保存完整记录

        return similar_workloads

    def calculate_similarity(self, wi: dict, wj: dict, common_solutions: set, beta) -> float:
        """
        wi/wj: Dict[config tuple] -> perf
        common_solutions: configs that appear in both workload-i and workload-j
        beta: used as fallback threshold for minimum comparison size
        """
        if len(common_solutions) < 2:
            return round(random.uniform(0, beta), 4)

        total_pairs = 0
        consistent_pairs = 0

        for sol1, sol2 in combinations(common_solutions, 2):
            p1_env1 = wi[sol1]
            p2_env1 = wi[sol2]
            p1_env2 = wj[sol1]
            p2_env2 = wj[sol2]

            total_pairs += 1
            if (p1_env1 > p2_env1) == (p1_env2 > p2_env2):
                consistent_pairs += 1

        similarity_score = consistent_pairs / total_pairs if total_pairs > 0 else 0
        return round(similarity_score, 4)

    def weighted_config_seeding(self, similar_workloads: dict, current_workload):
        """
        融合多个相似 workload 的 top 50% 配置，
        赋予鲁棒性、时间线和相似性权重，进行加权采样。
        """
        # Step 1: 时间顺序索引（按 similar_workloads 中的 key 排序）
        sorted_wls = sorted(similar_workloads.keys(), key=lambda wl: self.workloads.index(wl))
        timeline_index = {wl: (i + 1) / len(sorted_wls) for i, wl in enumerate(sorted_wls)}  # 1/k, 2/k, ..., 1.0

        # Step 2: 收集所有候选配置及其来源 workload 和 similarity
        cfg_to_workloads = defaultdict(set)  # cfg → set of workload names (用于鲁棒性)
        cfg_to_sim = defaultdict(list)  # cfg → list of similarity scores
        cfg_to_latest_time = {}

        # 首先获取所有 workload 的相似性
        sim_values = list(similar_workloads.items())  # [(wl1, sim1), (wl2, sim2), ...]
        sim_values_sorted = sorted(sim_values, key=lambda x: x[1], reverse=True)
        sim_rank = {wl: (len(sim_values_sorted) - i) / len(sim_values_sorted) for i, (wl, _) in
                    enumerate(sim_values_sorted)}
        # eg: top 1 → 1.0, second → 0.75, third → 0.5, last → 0.25 (when 4 workloads)

        for wl in similar_workloads:
            pop = self.history_optimized_population[wl]
            if not pop:
                continue

            reverse = self.optimization_goal == "maximum"
            sorted_pop = sorted(pop, key=lambda x: x[1], reverse=reverse)
            top_k = sorted_pop[:len(sorted_pop) // 2]

            for cfg, _ in top_k:
                cfg_t = tuple(cfg)
                cfg_to_workloads[cfg_t].add(wl)
                cfg_to_sim[cfg_t].append(sim_rank[wl])  # 添加相似性分数;用排名分数而不是直接相似性分数；
                cfg_to_latest_time[cfg_t] = timeline_index[wl]  # 记录该 config 的最后出现时间

        if not cfg_to_workloads:
            return self.config_manager.random_initialize_population(self.pop_size)

        # Step 3: 计算最终权重
        unique_cfgs = list(cfg_to_workloads.keys())
        weights = []

        for cfg in unique_cfgs:
            robustness_weight = len(cfg_to_workloads[cfg]) / len(similar_workloads)
            timeline_weight = cfg_to_latest_time[cfg]
            similarity_weight = sum(cfg_to_sim[cfg]) / len(cfg_to_sim[cfg])

            total_weight = robustness_weight + timeline_weight + similarity_weight
            weights.append(total_weight)

        weights = np.array(weights)

        # Step 4: 采样生成初始种群（改为根据权重排序）
        sorted_indices = np.argsort(weights)[::-1]  # 从大到小排序
        selected = [list(unique_cfgs[i]) for i in sorted_indices[:self.pop_size]]
        selected_set = {tuple(unique_cfgs[i]) for i in sorted_indices[:self.pop_size]}

        # 若候选不足，则补充不重复个体
        if len(selected) < self.pop_size:
            existing_set = selected_set
            needed = self.pop_size - len(selected)
            filler = self.config_manager.random_initialize_population(needed, existing_configs=existing_set)
            selected.extend(filler)

        '''
        根据概率来选
        # Step 5: 计算权重概率分布
        probs = weights / weights.sum()
        # Step 4: 采样生成初始种群
        selected = []
        selected_set = set()
        if len(unique_cfgs) >= self.pop_size:
            selected_idx = np.random.choice(len(unique_cfgs), size=self.pop_size, replace=False, p=probs)
            selected = [list(unique_cfgs[i]) for i in selected_idx]
            selected_set = {unique_cfgs[i] for i in selected_idx}
        else:
            selected = [list(cfg) for cfg in unique_cfgs]
            selected_set = set(unique_cfgs)
            remaining = self.config_manager.random_initialize_population(self.pop_size - len(selected))
            selected.extend(remaining)
        '''

        # ✅ Step 5: 保存 candidates.csv
        output_dir = f"results/run{self.run_id}/DLiSA-C/{self.system}/{current_workload}"
        os.makedirs(output_dir, exist_ok=True)

        rows = []
        for cfg, weight in zip(unique_cfgs, weights):
            label = 1 if cfg in selected_set else 0
            row = list(cfg) + [round(weight, 6), label]
            rows.append(row)

        config_header = self.logger.header[:-1]  # Exclude performance column
        df = pd.DataFrame(rows, columns=config_header + ["weight", "selected"])
        df.to_csv(os.path.join(output_dir, "purified_candidates.csv"), index=False)


        # ✅ 保存历史所有 workload 的个体，用于后续分析
        all_rows = []

        for wl, pop in self.history_optimized_population.items():
            if not pop:
                continue
            sorted_pop = sorted(pop, key=lambda x: x[1], reverse=(self.optimization_goal == "maximum"))
            top_k = sorted_pop[:len(sorted_pop)]
            for cfg, _ in top_k:
                row = list(cfg) + [wl]
                all_rows.append(row)

        # 写入 CSV 文件
        all_history_path = os.path.join(output_dir, "all_candidates.csv")
        config_header = self.logger.header[:-1]  # 移除最后的 perf 列
        df_all = pd.DataFrame(all_rows, columns=config_header + ["workload"])
        df_all.to_csv(all_history_path, index=False)

        return selected

    def save_similarity_records(self):
        os.makedirs(os.path.dirname(self.similarity_log_path), exist_ok=True)
        with open(self.similarity_log_path, 'w') as f:
            json.dump(self.similarity_records, f, indent=4)

