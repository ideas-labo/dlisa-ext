import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from utility import Utility


class PerfCorrelation:
    def __init__(self, system_name, pearson_threshold, kendall_threshold, spearman_threshold):
        self.system_path = os.path.join('datasets', system_name)
        self.system_name = system_name
        self.perf_dict = Utility.load_perf_data(self.system_path, self.system_name)
        self.pearson_threshold = pearson_threshold
        self.kendall_threshold = kendall_threshold
        self.spearman_threshold = spearman_threshold
        self.local_optima_overlap_heatmap_path = os.path.join(
            'landscape_analysis', system_name, 'landscape_visualization', 'heatmap')
        os.makedirs(self.local_optima_overlap_heatmap_path, exist_ok=True)

    def calculate_correlations(self):
        performance_df = pd.DataFrame(self.perf_dict)
        pearson_corr = performance_df.corr(method='pearson')
        kendall_corr = performance_df.corr(method='kendall')
        spearman_corr = performance_df.corr(method='spearman')
        return pearson_corr, kendall_corr, spearman_corr

    def correlation_visualization(self):
        pearson_corr, kendall_corr, spearman_corr = self.calculate_correlations()
        self._plot_correlation_matrix(kendall_corr, method_name="Kendall", threshold=self.kendall_threshold)
        self._plot_correlation_matrix(spearman_corr, method_name="Spearman", threshold=self.spearman_threshold)

    def _plot_correlation_matrix(self, corr_df, method_name="Kendall", threshold=0.39):
        workloads = list(corr_df.columns)
        num_workloads = len(workloads)

        fig, ax = plt.subplots(figsize=(10, 11))
        cmap = cm.RdBu_r
        norm = mcolors.Normalize(vmin=0, vmax=1)
        short_labels = [f"W{i + 1}" for i in range(num_workloads)]

        for i in range(num_workloads):
            for j in range(i + 1):
                val = corr_df.iloc[i, j]
                if i != j and val < threshold:
                    continue
                radius = 0.4
                color = cmap(norm(val))
                circle = plt.Circle((j + 0.5, i + 0.5), radius=radius,
                                    facecolor=color, alpha=0.75, edgecolor='black', linewidth=0.5)
                ax.add_patch(circle)
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha='center', va='center', fontsize=16, color='black')


        ax.set_xlim(0, num_workloads)
        ax.set_ylim(0, num_workloads)
        ax.set_xticks(np.arange(num_workloads) + 0.5)
        ax.set_yticks(np.arange(num_workloads) + 0.5)
        ax.set_xticklabels(short_labels, fontsize=14)
        ax.set_yticklabels(short_labels, fontsize=14)
        ax.invert_yaxis()

        for i in range(num_workloads):
            for j in range(i + 1):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='gray', linewidth=0.5))


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.06, pad=0.05)
        cbar.set_label(f"{method_name} Correlation", fontsize=22)


        print(f"\n📘 Workload mapping ({method_name}):")
        for i, wl in enumerate(workloads):
            print(f"  W{i + 1}: {wl}")

        output_path = os.path.join(self.local_optima_overlap_heatmap_path,
                                   f"{method_name.lower()}_correlation_triangle_circles.pdf")
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"✅: {method_name} correlation → {output_path}")
