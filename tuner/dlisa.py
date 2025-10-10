import os
import random
from itertools import combinations
import json
import numpy as np
import pandas as pd

from utils.evolutionary_planning import EvolutionaryPlanner
from utils.config_manager import ConfigManager
from utils.logger import Logger
from collections import defaultdict, Counter


class DLiSATuner:
    def __init__(self, system, workloads, run_id, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback=None):
        self.logger = None
        self.config_manager = None
        self.system = system
        self.workloads = workloads
        self.run_id = run_id
        self.optimization_goal = optimization_goal
        self.initial_seeds = initial_seeds
        self.common_seeds = common_seeds
        self.fallback = fallback  # Fallback strategy for configurations not found in the dataset

        self.max_generation = max_generation
        self.pop_size = pop_size
        self.history_optimized_population = {}

        self.similarity_log_path = f"results/run{self.run_id}/DLiSA/{self.system}/similarity.json"
        self.average_similarity_record = {}


    def run(self):
        self.config_manager = ConfigManager(system=self.system, optimization_goal=self.optimization_goal)

        for workload_id, workload_name in enumerate(self.workloads):
            print(f"\n[DLiSA | RUN {self.run_id}] >>> Workload {workload_id + 1}/{len(self.workloads)}: {workload_name}")
            self.config_manager.set_workload(workload_name)

            header = self.config_manager.get_config_header()
            self.logger = Logger(
                base_dir=f"results/run{self.run_id}",
                algorithm="DLiSA",
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
                init_population = self.generate_seeds_by_dlisa(workload_name)

            planner = EvolutionaryPlanner(
                config_manager=self.config_manager,
                init_population=init_population,
                max_generation=self.max_generation,
                mutation_rate=0.1,
                crossover_rate=0.9,
                common_seeds=self.common_seeds,
                pop_size=self.pop_size,
                logger=self.logger,
                fallback=self.fallback
            )
            optimized_pop = planner.evolutionary_search_config()
            self.history_optimized_population[workload_name] = optimized_pop

    def generate_seeds_by_dlisa(self, current_workload):
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
            average_similarity = self.ranking_workload_similarity_analysis()
            self.average_similarity_record[current_workload] = average_similarity
            self.save_similarity_record()  # 保存到文件

            if average_similarity >= 0.3:
                init_population = self.weighted_config_seeding(current_workload)
            else:
                init_population = self.config_manager.random_initialize_population(self.pop_size)

        return init_population

    def ranking_workload_similarity_analysis(self) -> float:

        workload_names = list(self.history_optimized_population.keys())

        workload_to_perf_dict = {
            wl: {tuple(cfg): perf for cfg, perf in pop}
            for wl, pop in self.history_optimized_population.items()
        }

        similarities = []
        for i in range(len(workload_names) - 1):
            wl1 = workload_names[i]
            wl2 = workload_names[i + 1]
            dict1 = workload_to_perf_dict[wl1]
            dict2 = workload_to_perf_dict[wl2]

            common_configs = set(dict1.keys()) & set(dict2.keys())
            sim = self.calculate_similarity(dict1, dict2, common_configs, beta=0.3)
            similarities.append(sim)

            if not similarities:
                return 0.0
        return round(sum(similarities) / len(similarities), 4)

    def calculate_similarity(self, wi: dict, wj: dict, common_solutions: set, beta: float = 0.3) -> float:
        """
        wi/wj: Dict[config tuple] -> perf
        common_solutions: configs that appear in both workload-i and workload-j
        beta: used as fallback threshold for minimum comparison size
        """
        if len(common_solutions) < 2:
            return beta  # round(random.uniform(0, 0.6), 4)

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

    def weighted_config_seeding(self, current_workload):
        """
        聚合历史 workload 的 top 50% 配置作为候选集，
        基于鲁棒性 + 时间权重加权采样种群。
        """
        all_candidates = []  # List[Tuple[config_tuple]]
        workload_order = list(self.history_optimized_population.keys())
        workload_count = len(workload_order)

        for i, wl in enumerate(workload_order):
            pop = self.history_optimized_population[wl]
            if not pop:
                continue
            sorted_pop = sorted(pop, key=lambda x: x[1], reverse=(self.optimization_goal == "maximum"))
            top_k = sorted_pop[:len(sorted_pop) // 2]
            for cfg, _ in top_k:
                all_candidates.append((tuple(cfg), i))  # 加上所属的时间 index

        if not all_candidates:
            return self.config_manager.random_initialize_population(self.pop_size)

        # Step 1: 统计每个配置出现的次数（鲁棒性权重） & 最后出现在哪个 workload（时间权重）
        freq_counter = Counter()
        latest_time = defaultdict(int)

        for cfg, t in all_candidates:
            freq_counter[cfg] += 1
            latest_time[cfg] = max(latest_time[cfg], t)

        # Step 2: 计算权重
        compound_weights = []
        configs = []

        for cfg in freq_counter:
            robustness_weight = freq_counter[cfg] / workload_count
            timeliness_weight = (1 + latest_time[cfg]) / workload_count
            compound = robustness_weight + timeliness_weight
            compound_weights.append(compound)
            configs.append(cfg)

        compound_weights = np.array(compound_weights)
        probabilities = compound_weights / compound_weights.sum()

        # Step 3: 采样生成初始种群
        selected = []
        selected_set = set()

        if len(configs) >= self.pop_size:
            selected_indices = np.random.choice(len(configs), size=self.pop_size, replace=False, p=probabilities)
            selected = [list(configs[i]) for i in selected_indices]
            selected_set = {configs[i] for i in selected_indices}
            existing_set = {tuple(cfg) for cfg in selected_set}
        else:
            selected = [list(cfg) for cfg in configs]
            selected_set = set(configs)
            existing_set = {tuple(cfg) for cfg in configs}
            needed = self.pop_size - len(selected)
            filler = self.config_manager.random_initialize_population(needed, existing_configs=existing_set)
            selected.extend(filler)

        # Step 4: 写入 candidates.csv
        output_path = f"results/run{self.run_id}/DLiSA/{self.system}/{current_workload}"
        os.makedirs(output_path, exist_ok=True)

        records = []
        for cfg, weight in zip(configs, compound_weights):
            label = 1 if cfg in selected_set else 0
            record = list(cfg) + [round(weight, 6), label]
            records.append(record)

        config_header = self.logger.header[:-1]  # Exclude performance column
        df = pd.DataFrame(records, columns=config_header + ["weight", "selected"])
        df.to_csv(os.path.join(output_path, "purified_candidates.csv"), index=False)

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
        all_history_path = os.path.join(output_path, "all_candidates.csv")
        config_header = self.logger.header[:-1]  # 移除最后的 perf 列
        df_all = pd.DataFrame(all_rows, columns=config_header + ["workload"])
        df_all.to_csv(all_history_path, index=False)


        return selected

    def save_similarity_record(self):
        os.makedirs(os.path.dirname(self.similarity_log_path), exist_ok=True)
        with open(self.similarity_log_path, 'w') as f:
            json.dump(self.average_similarity_record, f, indent=4)



