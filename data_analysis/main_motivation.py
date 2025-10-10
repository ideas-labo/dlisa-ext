import os
from perf_analyzer import PerfAnalyzer
from landscape_visualizer import LandscapeVisualizer
from budget_perf_analyzer import BudgetPerfAnalyzer
from perf_correlation import PerfCorrelation


def main():

    system_names = ['batik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    system_names = ['kanzi']

    for system_name in system_names:


        # analyzer = BudgetPerfAnalyzer(system_name)
        # analyzer.analyze_budget_vs_full(sample_ratio=0.2)


        landscape_visualizer = LandscapeVisualizer(system_name)
        # # #
        config_2d_dict = landscape_visualizer.transform_configs_by_mds()
        # config_2d_dict = landscape_visualizer.transform_configs_by_tsne()
        # # #
        # landscape_visualizer.performance_distribution_visualization(config_2d_dict)
        # # #
        #
        landscape_visualizer.full_landscape_contour_visualization(config_2d_dict)
        # landscape_visualizer.full_landscape_visualization(config_2d_dict)
        # landscape_visualizer.full_landscape_visualization_3d_normalization(config_2d_dict)
        # #
        # landscape_visualizer.top_index_overlap_3d_visualization(config_2d_dict)

        """-----------------replicate the correlation of the original paper--------------"""
        pearson_threshold = 0.39
        kendall_threshold = 0.39
        spearman_threshold = 0.39
        # perf_analyzer = PerfAnalyzer(system_name, pearson_threshold, kendall_threshold, spearman_threshold)
        # perf_analyzer.correlation_visualization()
        # analyzer = PerfCorrelation(system_name, pearson_threshold, kendall_threshold, spearman_threshold)
        # analyzer.correlation_visualization()


if __name__ == "__main__":
    main()
