import os
import random
import pandas as pd
from typing import List, Tuple
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.preprocessing import OrdinalEncoder


class ConfigManager:
    def __init__(self, system: str, optimization_goal: str = "minimum"):
        self.balltree = None
        self.encoded_configs = None
        self.encoder = None
        self.config_perf_dict = None
        self.system = system
        self.optimization_goal = optimization_goal
        self.config_dim = None
        self.config_space = []
        self.value_domains = []
        self.init_config_space()

        self.workload_name = None
        self.data_df = None
        self.worst_perf = None


        self.feature_types = []
        self.norm_factors = []



    def init_config_space(self):
        system_path = os.path.join("datasets", self.system)
        csv_files = [f for f in os.listdir(system_path) if f.endswith(".csv")]

        if not csv_files:
            raise FileNotFoundError(f"[ConfigManager] No workload CSV found in {system_path}")

        sample_df = pd.read_csv(os.path.join(system_path, csv_files[0]))
        config_df = sample_df.iloc[:, :-1]
        self.config_dim = config_df.shape[1]

        self.config_space = config_df.drop_duplicates().values.tolist()

        self.value_domains = []
        for col in config_df.columns:
            unique_vals = config_df[col].unique().tolist()
            self.value_domains.append(unique_vals)

    def set_workload(self, workload_name: str):
        self.workload_name = workload_name
        path = os.path.join("datasets", self.system, f"{workload_name}.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"[ConfigManager] Workload CSV not found: {path}")

        self.data_df = pd.read_csv(path)
        self.precompute_worst_perf()
        self.config_perf_dict = {
            tuple(row[:-1]): row[-1]
            for row in self.data_df.values
        }
        self.build_balltree()
        self.detect_feature_types()


    def precompute_worst_perf(self):
        perf_values = self.data_df.iloc[:, -1]
        if self.optimization_goal == "minimum":
            self.worst_perf = perf_values.max()
        elif self.optimization_goal == "maximum":
            self.worst_perf = perf_values.min()
        else:
            raise ValueError(f"Unknown optimization goal: {self.optimization_goal}")

    def random_initialize_common_seeds(self, seed_size):
        seen = set()
        seeds = []
        attempts = 0
        max_attempts = seed_size * 10

        while len(seeds) < seed_size and attempts < max_attempts:
            config = [random.choice(values) for values in self.value_domains]
            config_tuple = tuple(config)
            if config_tuple not in seen:
                seen.add(config_tuple)
                seeds.append(config)
            attempts += 1

        if len(seeds) < seed_size:
            print(f"[WARN] Only generated {len(seeds)} unique configs out of requested {seed_size}")

        return seeds

    def random_initialize_population(self, pop_size, existing_configs=None):
        """
        Returns a mixed population:
        - Half randomly sampled from config_space
        - Half randomly generated from value_domains
        Ensures no duplication with existing_configs (if provided).
        """
        if existing_configs is None:
            existing_configs = set()
        else:
            existing_configs = set(tuple(cfg) for cfg in existing_configs)

        size_from_tabular = pop_size // 2  # 1/2 from config_space, 1/2 generated, which is ensuring there are validate configs
        seen = set(existing_configs)

        # 从 config_space 表格中随机采样，不重复
        sampled_from_csv = []
        available_from_csv = [cfg for cfg in self.config_space if tuple(cfg) not in seen]
        if available_from_csv:
            sampled_from_csv = random.sample(available_from_csv, min(size_from_tabular, len(available_from_csv)))
            seen.update(tuple(cfg) for cfg in sampled_from_csv)

        # 第二半：从 value_domains 中构造随机解
        generated = []
        attempts = 0
        max_attempts = pop_size * 10

        while len(generated) < (pop_size - len(sampled_from_csv)) and attempts < max_attempts:
            config = [random.choice(values) for values in self.value_domains]
            config_tuple = tuple(config)
            if config_tuple not in seen:
                seen.add(config_tuple)
                generated.append(config)
            attempts += 1

        population = sampled_from_csv + generated

        if len(population) < pop_size:
            print(f"[WARN] Only generated {len(population)} unique configs out of requested {pop_size}")

        return population

    def maxmin_sampling_hamming(self, k: int) -> List[List]:
        # 去重（保持原始类型）
        configs = list({tuple(cfg): cfg for cfg in self.config_space}.values())

        if len(configs) <= k:
            return configs

        # 用于距离计算：将配置转为字符串数组（用于比较）
        configs_str = [list(map(str, cfg)) for cfg in configs]
        configs_str = [[self.to_str_clean(x) for x in cfg] for cfg in configs]
        config_space_str = np.array(configs_str)
        n = len(config_space_str)

        selected_indices = [np.random.randint(0, n)]

        while len(selected_indices) < k:
            selected_configs = config_space_str[selected_indices]
            remaining_indices = list(set(range(n)) - set(selected_indices))

            distances = []
            for idx in remaining_indices:
                candidate = config_space_str[idx]
                min_dist = np.min([
                    self.hamming_distance(candidate, sel) for sel in selected_configs
                ])
                distances.append((idx, min_dist))

            next_idx = max(distances, key=lambda x: x[1])[0]
            selected_indices.append(next_idx)

        # 最终返回原始类型的配置
        return [configs[idx] for idx in selected_indices]

    @staticmethod
    def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
        return np.sum(a != b)

    def get_config_header(self):
        return self.data_df.columns[:].tolist()

    def build_balltree(self, sklearn=None):
        # 将 config_space 转为字符串，再编码为整数数组以构建 BallTree
        self.encoder = OrdinalEncoder()
        str_configs = [[self.to_str_clean(v) for v in cfg] for cfg in self.config_perf_dict.keys()]
        self.encoded_configs = self.encoder.fit_transform(str_configs)  # shape: (n_samples, n_features)
        self.balltree = BallTree(self.encoded_configs, metric='hamming')

    # def evaluate(self, config_list: List[List[float]], fallback='worst') -> List[Tuple[List[float], float]]:
    #     if self.data_df is None or self.config_perf_dict is None:
    #         raise RuntimeError("Workload not set or config dict not built.")
    #
    #     results = []
    #     for config in config_list:
    #         # perf = self.config_perf_dict.get(tuple(config), self.worst_perf)
    #         key = tuple(config)
    #         if key in self.config_perf_dict:
    #             perf = self.config_perf_dict[key]
    #         elif fallback == 'nearest':
    #             # encode current config, find nearest in balltree
    #             str_config = [self.to_str_clean(v) for v in config]
    #             encoded = self.encoder.transform([str_config])
    #             dist, idx = self.balltree.query(encoded, k=1)
    #             nearest_cfg = list(self.config_perf_dict.keys())[idx[0][0]]
    #             perf = self.config_perf_dict[nearest_cfg]
    #         elif fallback == 'worst':
    #             perf = self.worst_perf
    #
    #         results.append((config, perf))
    #     return results



    def evaluate(self, config_list: List[List[float]], fallback='worst') -> List[Tuple[List[float], float]]:
        if self.data_df is None or self.config_perf_dict is None:
            raise RuntimeError("Workload not set or config dict not built.")

        results = []
        for config in config_list:
            key = tuple(config)
            if key in self.config_perf_dict:
                perf = self.config_perf_dict[key]
            elif fallback == 'nearest':
                nearest_cfg = self.find_nearest_by_mixed_distance(config)
                perf = self.config_perf_dict[nearest_cfg]
            elif fallback == 'worst':
                perf = self.worst_perf + np.random.uniform(0.001, 0.009)
            else:
                raise ValueError(f"Unknown fallback strategy: {fallback}")

            results.append((config, perf))
        return results

    def to_str_clean(self, val):
        return str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)

    def detect_feature_types(self):
        """自动判断每列是 categorical 还是 numerical，并设置归一化因子"""
        sample_cfgs = list(self.config_perf_dict.keys())
        zipped = list(zip(*sample_cfgs))

        self.feature_types = []
        self.norm_factors = []

        for values in zipped:
            try:
                float_vals = list(map(float, values))
                unique_vals = set(float_vals)
            except:
                # 转不了 float 的一律 categorical
                self.feature_types.append("categorical")
                self.norm_factors.append(1)
                continue

            # 只有两个取值且是 0/1，当作 categorical（二值布尔）
            if unique_vals <= {0.0, 1.0} and len(unique_vals) <= 2:
                self.feature_types.append("categorical")
                self.norm_factors.append(1)
            else:
                # 其余只要能 float 的，都统一当成 numerical
                self.feature_types.append("numerical")
                max_val = max(float_vals)
                min_val = min(float_vals)
                norm = max_val - min_val
                self.norm_factors.append(norm if norm != 0 else 1)

    def mixed_distance(self, cfg1, cfg2):
        """支持 categorical 和 numerical 混合的距离计算"""
        dist = 0
        for i, (v1, v2) in enumerate(zip(cfg1, cfg2)):
            if self.feature_types[i] == "categorical":
                dist += int(v1 != v2)
            elif self.feature_types[i] == "numerical":
                dist += abs(float(v1) - float(v2)) / self.norm_factors[i]
        return dist / len(cfg1)



    def find_nearest_by_mixed_distance(self, config):
        """暴力搜索找最近的配置（基于混合距离）"""
        min_dist = float("inf")
        nearest = None
        for candidate in self.config_perf_dict.keys():
            d = self.mixed_distance(config, candidate)
            if d < min_dist:
                min_dist = d
                nearest = candidate
        return nearest
