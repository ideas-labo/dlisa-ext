import os
import pandas as pd
from typing import List, Tuple
import json


class Logger:
    def __init__(self, base_dir: str, algorithm: str, system: str, workload: str, header: List[str]):
        self.header = header  # n-1 config names + 1 perf name

        # 构造完整路径
        self.log_dir = os.path.join(base_dir, algorithm, system, workload)
        os.makedirs(self.log_dir, exist_ok=True)

    def log_generation(self, generation: int, population: List[Tuple[List, float]]):
        """
        保存当前代的种群为 gen{n}.csv
        """
        filename = os.path.join(self.log_dir, f"gen{generation}.csv")
        self._write_csv(filename, population)

    def log_all_results(self, all_results: List[Tuple[List, float]]):
        """
        保存整个搜索过程中的所有评估配置与性能
        """
        filename = os.path.join(self.log_dir, "all_results.csv")
        self._write_csv(filename, all_results)

    def _write_csv(self, filepath: str, data: List[Tuple[List, float]]):
        """
        通用写入器：配置 + 性能 → dataframe + 表头
        """
        rows = [cfg + [perf] for cfg, perf in data]
        df = pd.DataFrame(rows, columns=self.header)
        df.to_csv(filepath, index=False)

