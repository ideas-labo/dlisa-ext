import os
import random
import numpy as np
from matplotlib import pyplot as plt

from utility import Utility


class BudgetPerfAnalyzer:
    def __init__(self, system_name):
        self.system_name = system_name
        self.system_path = os.path.join('dataset', system_name)
        self.perf_dict = Utility.load_perf_data(self.system_path, self.system_name)
        self.config_dict = Utility.load_config_data(self.system_path, self.system_name)

    def analyze_budget_vs_full(self, sample_ratio=0.2, repeat=10, visualize=True):
        all_results = {}
        short_labels = []
        improvements = []
        label_mapping = {}

        for i, (workload_key, perf_series) in enumerate(self.perf_dict.items(), start=1):
            if perf_series.empty:
                print(f"[Warning] Empty performance data for: {workload_key}")
                continue

            improvement_list = []
            if self.system_name == 'mysql' or self.system_name == 'postgresql':
                global_best = perf_series.max()

                for _ in range(repeat):
                    sampled_series = perf_series.sample(frac=sample_ratio)
                    sampled_best = sampled_series.max()

                    if sampled_best == 0:
                        improvement = float('inf') if global_best > 0 else 0.0
                    else:
                        improvement = (global_best - sampled_best) / sampled_best * 100
                    improvement_list.append(improvement)
            elif self.system_name == 'gcc' or self.system_name == 'clang':
                global_best = perf_series.min()

                for _ in range(repeat):
                    sampled_series = perf_series.sample(frac=sample_ratio)
                    sampled_best = sampled_series.min()

                    if sampled_best == 0:
                        improvement = float('inf') if global_best > 0 else 0.0
                    else:
                        improvement = (sampled_best  - global_best) / sampled_best * 100
                    improvement_list.append(improvement)

            avg_improvement = np.mean(improvement_list)
            std_improvement = np.std(improvement_list)

            short_label = f"W{i}"
            label_mapping[short_label] = workload_key
            short_labels.append(short_label)
            improvements.append(avg_improvement)

            print(
                f"{workload_key}: avg improvement {avg_improvement:.2f}% ± {std_improvement:.2f}% over {repeat} trials")
            all_results[workload_key] = {
                "global_best": global_best,
                "avg_improvement_percent": avg_improvement,
                "std_improvement_percent": std_improvement,
                "raw_improvements": improvement_list
            }

        if visualize:
            self.plot_improvement_bar(short_labels, improvements, label_mapping)

        return all_results

    def plot_improvement_bar(self, labels, improvements, label_mapping):
        plt.figure(figsize=(10, 8))
        bars = plt.bar(labels, improvements)

        # Optional: annotate values on top of bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{imp:.1f}%", ha='center', va='bottom', fontsize=12)

        plt.xlabel("Workloads", fontsize=12)

        plt.ylabel("Performance Improvement (%)", fontsize=12)
        plt.title(f"Improvement when going from 20% budget to full budget ({self.system_name})", fontsize=12)
        plt.tight_layout()

        # Add mapping info as a text box (for long names)
        label_text = "\n".join([f"{k}: {v}" for k, v in label_mapping.items()])
        plt.gcf().text(1.02, 0.5, label_text, fontsize=14, va='center', transform=plt.gcf().transFigure)

        plt.show()

