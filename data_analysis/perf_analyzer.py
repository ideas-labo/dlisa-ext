import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from utility import Utility


class PerfAnalyzer:
    def __init__(self, system_name, pearson_threshold, kendall_threshold, spearman_threshold):
        self.system_path = os.path.join('datasets', system_name)
        self.system_name = system_name
        self.perf_dict = Utility.load_perf_data(self.system_path, self.system_name)
        self.pearson_threshold = pearson_threshold
        self.kendall_threshold = kendall_threshold
        self.spearman_threshold = spearman_threshold
        self.local_optima_overlap_heatmap_path = os.path.join('landscape_analysis', system_name, 'landscape_visualization', 'heatmap')
        if not os.path.exists(self.local_optima_overlap_heatmap_path):
            os.makedirs(self.local_optima_overlap_heatmap_path)

    def calculate_correlations(self):
        """calculate the Pearson and Kendall correlation"""
        performance_df = pd.DataFrame(self.perf_dict)
        pearson_corr = performance_df.corr(method='pearson')
        kendall_corr = performance_df.corr(method='kendall')
        spearman_corr = performance_df.corr(method='spearman')
        return pearson_corr, kendall_corr, spearman_corr

    def plot_heatmaps_with_threshold(self, pearson_corr, kendall_corr, spearman_corr, system_name,
                                     pearson_threshold=0.6, kendall_threshold=0.6, spearman_threshold=0.6):
        """Plot heatmaps for correlations above thresholds"""
        pearson_mask = self._sanitize_mask(np.abs(pearson_corr) <= self.pearson_threshold)
        kendall_mask = self._sanitize_mask(np.abs(kendall_corr) <= self.kendall_threshold)
        spearman_mask = self._sanitize_mask(np.abs(spearman_corr) <= self.spearman_threshold)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        self.plot_single_heatmap(axes[0], kendall_corr, kendall_mask, f"Kendall's τ (τ > {kendall_threshold})")
        self.plot_single_heatmap(axes[1], spearman_corr, spearman_mask, f"Spearman's τ (τ > {spearman_threshold})")

        fig.suptitle(f"Correlation Analysis for {system_name}")
        # plt.tight_layout()
        file_path = os.path.join(self.local_optima_overlap_heatmap_path, 'pearson_kendall_spearman_correlation.png')
        plt.savefig(file_path, dpi=720)
        plt.close(fig)

    def _sanitize_mask(self, mask):
        mask = mask.copy()
        for i in range(mask.shape[0]):
            if mask.iloc[i].all():
                mask.iat[i, i] = False
            if mask.iloc[:, i].all():
                mask.iat[i, i] = False
        return mask

    def plot_single_heatmap(self, ax, data, mask, title, vmin=0, vmax=1, cmap='coolwarm'):
        sns.heatmap(data, mask=mask, cmap=cmap, center=0.5,
                    annot=True, cbar=True, ax=ax, vmin=vmin, vmax=vmax,
                    linewidths=0, linecolor='white',
                    square=True)

        ax.set_facecolor((0, 0, 0, 0))

        ax.set_title(title)

    def correlation_visualization(self):
        pearson_corr, kendall_corr, spearman_corr = self.calculate_correlations()
        self.plot_heatmaps_with_threshold(pearson_corr, kendall_corr, spearman_corr, self.system_name, self.pearson_threshold,
                                          self.kendall_threshold, self.spearman_threshold)
