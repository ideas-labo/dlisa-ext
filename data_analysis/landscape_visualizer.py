import os
import re
import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
import seaborn as sns
from utility import Utility
import pandas as pd
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
import random
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge, Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
from sklearn.manifold import TSNE



class LandscapeVisualizer:
    def __init__(self, system_name):
        self.system_path = os.path.join('datasets', system_name)
        self.system_path_2d = os.path.join('datasets', system_name, 'workload_2d')
        self.system_name = system_name
        # self.perf_dict = Utility.load_perf_data(self.system_path, self.system_name)
        # self.config_dict = Utility.load_config_data(self.system_path, self.system_name)

        self.config_dict, self.perf_dict = Utility.load_config_and_perf_data(self.system_path, self.system_name)
        self.full_landscape_path = os.path.join('landscape_analysis', system_name, 'landscape_visualization', 'full_landscape')
        self.local_optima_overlap_heatmap_path = os.path.join('landscape_analysis', system_name, 'landscape_visualization', 'heatmap')
        self.top_5_percent = os.path.join('landscape_analysis', system_name, 'landscape_visualization', 'top_5_percent')
        self.perf_distribution_path = os.path.join('landscape_analysis', system_name, 'landscape_visualization', 'perf_distribution')
        if not os.path.exists(self.full_landscape_path):
            os.makedirs(self.full_landscape_path)
        if not os.path.exists(self.local_optima_overlap_heatmap_path):
            os.makedirs(self.local_optima_overlap_heatmap_path)
        if not os.path.exists(self.top_5_percent):
            os.makedirs(self.top_5_percent)
        if not os.path.exists(self.system_path_2d):
            os.makedirs(self.system_path_2d)
        if not os.path.exists(self.perf_distribution_path):
            os.makedirs(self.perf_distribution_path)



    def transform_configs_by_mds(self, n_components=2):
        """Conduct MDS reduction on all workloads for target system"""
        config_2d_dict = {}
        for workload, config_data in self.config_dict.items():
            file_path = os.path.join(self.system_path_2d, f"{workload}_2d.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                config_2d_dict[workload] = df[['MDS_Dim1', 'MDS_Dim2']].values
            else:
                # if config_2d_dict exist, retrieve it for saving time
                # as the config space is the same for all workload under one system, only get once
                if config_2d_dict:
                    config_2d = list(config_2d_dict.values())[0]
                    config_2d_dict[workload] = config_2d
                else:
                    # Normalization
                    config_data = pd.DataFrame(config_data)
                    preprocessed_data = self.preprocess_data(config_data)
                    mds = MDS(n_components=n_components, random_state=0)
                    config_2d = mds.fit_transform(preprocessed_data)
                    config_2d_dict[workload] = config_2d

                # logging the reduced data to csv
                df_2d = pd.DataFrame(config_2d, columns=['MDS_Dim1', 'MDS_Dim2'])
                df_2d['performance'] = self.perf_dict[workload]
                df_2d.to_csv(file_path, index=False)

        return config_2d_dict

    def transform_configs_by_tsne(
            self,
            n_components=2,
            perplexity=30,
            learning_rate='auto',
            n_iter=1000,
            metric='euclidean',
            init='pca',
            random_state=0,
            suffix='tsne'  # distinct MDS and TSNE
    ):
        """Use t-SNE to reduce config space for all workloads (2D by default)."""
        config_2d_dict = {}
        for workload, config_data in self.config_dict.items():
            file_path = os.path.join(self.system_path_2d, f"{workload}_{suffix}_{n_components}d.csv")

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                config_2d_dict[workload] = df[[f'TSNE_Dim{i + 1}' for i in range(n_components)]].values
                continue

            if config_2d_dict:
                coords = list(config_2d_dict.values())[0]
                config_2d_dict[workload] = coords
            else:
                config_df = pd.DataFrame(config_data)
                X = self.preprocess_data(config_df)

                n_samples = X.shape[0]
                eff_perplexity = min(perplexity, max(5, n_samples // 3)) if n_samples <= perplexity else perplexity

                tsne = TSNE(
                    n_components=n_components,
                    perplexity=eff_perplexity,
                    learning_rate=learning_rate,
                    n_iter=n_iter,
                    metric=metric,
                    init=init,
                    random_state=random_state,
                    verbose=0
                )
                coords = tsne.fit_transform(X)
                config_2d_dict[workload] = coords

            dim_cols = [f'TSNE_Dim{i + 1}' for i in range(n_components)]
            df_out = pd.DataFrame(config_2d_dict[workload], columns=dim_cols)
            df_out['performance'] = self.perf_dict[workload]
            os.makedirs(self.system_path_2d, exist_ok=True)
            df_out.to_csv(file_path, index=False)

        return config_2d_dict

    @staticmethod
    def preprocess_data(config_data):
        """Perform normalization for numeric columns and encode non-numeric categorical columns."""
        for col in config_data.columns:
            unique_values = config_data[col].dropna().unique()

            if set(unique_values).issubset({0, 1}):
                continue


            if not pd.api.types.is_numeric_dtype(config_data[col]):
                sorted_vals = sorted(unique_values)
                mapping = {val: idx for idx, val in enumerate(sorted_vals)}
                config_data[col] = config_data[col].map(mapping)

        non_binary_numeric_cols = [
            col for col in config_data.columns
            if pd.api.types.is_numeric_dtype(config_data[col]) and not set(config_data[col].dropna().unique()).issubset(
                {0, 1})
        ]
        if non_binary_numeric_cols:
            scaler = MinMaxScaler()
            config_data[non_binary_numeric_cols] = scaler.fit_transform(config_data[non_binary_numeric_cols])

        return config_data

    def full_landscape_visualization(self, config_2d_dict):
        """
        Visualize the complete landscape of each workload, marking all global optima.
        :param config_2d_dict: 2D configurations
        """
        for workload, config_2d in config_2d_dict.items():
            performance = self.perf_dict[workload]

            # Find global optima based on system name
            if self.system_name in ['httpd', 'tomcat', 'mysql', 'postgresql', 'mysql_sampling']:
                # For h2, higher performance is better (maximization)
                global_opt_indices = np.where(performance == np.max(performance))[0]
            else:
                # For other systems, lower performance is better (minimization)
                global_opt_indices = np.where(performance == np.min(performance))[0]
                performance = -performance  # Invert for better visualization

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot for the whole landscape
            ax.scatter(config_2d[:, 0], config_2d[:, 1], performance, c=performance, cmap='Spectral', marker='o')

            # Mark each global optimum with a black pentagram
            for global_opt_index in global_opt_indices:
                global_opt_config = config_2d[global_opt_index]
                global_opt_perf = performance[global_opt_index]
                ax.scatter(global_opt_config[0], global_opt_config[1], global_opt_perf, color='black', marker='*',
                           s=50,
                           label='Global Optimum' if global_opt_index == global_opt_indices[0] else "")

            # Add color bar
            # fig.colorbar(sc, ax=ax, label='Performance')

            ax.set_xlabel('MDS-1', fontsize=18)
            ax.set_ylabel('MDS-2', fontsize=18)
            # ax.set_zlim(0.04, 0.15)
            if self.system_name == 'httpd' or self.system_name == 'tomcat' or self.system_name == 'mysql' or self.system_name == 'postgresql' or self.system_name == 'mysql_sampling':
                ax.set_zlabel('Throughput', fontsize=18)
            else:
                ax.set_zlabel('Runtime', fontsize=18)
            ax.set_title(f'({workload}) of System ({self.system_name})')
            plt.show()
            file_path = os.path.join(self.full_landscape_path, f'{workload}_landscape.pdf')
            plt.savefig(file_path, bbox_inches='tight', format='pdf')
            # file_path = os.path.join(self.full_landscape_path, f'{workload}_landscape.png')
            # plt.savefig(file_path, bbox_inches='tight', dpi=1024)
            # plt.yscale("log")
            # plt.show()
            plt.close()

    def full_landscape_contour_visualization(self, config_2d_dict):
        """
        Visualize the complete landscape of each workload using 2D contour plots.
        :param config_2d_dict: Dictionary of workload to 2D configuration points.
        """

        # order by workload name，making sure W1...Wn
        sorted_workloads = sorted(config_2d_dict.keys())

        for idx, workload in enumerate(sorted_workloads, start=1):
            config_2d = config_2d_dict[workload]
            performance = self.perf_dict[workload]

            wi_name = workload #f"W{idx}"
            print(f"{workload} -> {wi_name}")

            # Determine global optima based on system type
            if self.system_name in ['h2', 'httpd', 'tomcat', 'mysql', 'postgresql', 'mysql_sampling', 'lighttpd']:
                global_opt_indices = np.where(performance == np.max(performance))[0]
                # For top 5% selection
                threshold = np.percentile(performance, 95)
                top5_indices = np.where(performance >= threshold)[0]
            else:
                global_opt_indices = np.where(performance == np.min(performance))[0]
                threshold = np.percentile(performance, 5)
                top5_indices = np.where(performance <= threshold)[0]

            x = config_2d[:, 0]
            y = config_2d[:, 1]
            z = performance.copy()

            valid_mask = z != 0
            z_valid = z[valid_mask]

            z_min = np.min(z_valid)
            z_max = np.max(z_valid)
            z_norm = np.full_like(z, np.nan, dtype=np.float32)
            z_norm[valid_mask] = (z_valid - z_min) / (z_max - z_min + 1e-8)

            fig, ax = plt.subplots(figsize=(6, 5))

            contour = ax.tricontourf(x[valid_mask], y[valid_mask], z_norm[valid_mask], levels=10, cmap='RdBu_r')

            # Top 5% points
            ax.plot(x[top5_indices], y[top5_indices], 'o',
                    markerfacecolor='green', markersize=8,
                    markeredgecolor='black', label='Top 5% high-performing configurations')

            # Colorbar
            cbar = plt.colorbar(contour, ax=ax)
            if self.system_name in ['httpd', 'tomcat', 'mysql', 'postgresql', 'mysql_sampling']:
                cbar.set_label('Normalized Throughput', fontsize=16)
            else:
                cbar.set_label('Normalized Runtime', fontsize=16)

            ax.set_xlabel('#D1', fontsize=14)
            ax.set_ylabel('#D2', fontsize=14)
            # ax.set_title(f'{wi_name} of {self.system_name.upper()}', fontsize=18)
            ax.legend()

            file_path = os.path.join(self.full_landscape_path, f'{wi_name}_contour_landscape.pdf')
            plt.savefig(file_path, bbox_inches='tight', format='pdf')
            plt.close()

    def full_landscape_visualization_3d_normalization(self, config_2d_dict):
        """
        Visualize the complete landscape of each workload as a 3D surface with normalized performance.
        :param config_2d_dict: 2D configurations
        """
        for workload, config_2d in config_2d_dict.items():
            performance = self.perf_dict[workload]


            if self.system_name in ['h2', 'mysql']:
                global_opt_indices = np.where(performance == np.max(performance))[0]  # Maximization
            else:
                global_opt_indices = np.where(performance == np.min(performance))[0]  # Minimization
                # performance = -performance  # Invert for better visualization



            x = config_2d[:, 0]
            y = config_2d[:, 1]
            z = performance

            # insert data
            grid_x, grid_y = np.meshgrid(
                np.linspace(x.min(), x.max(), 20),  # resolution
                np.linspace(y.min(), y.max(), 20)
            )
            grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')  # nearest linear cubic
            # rbf_func = Rbf(x, y, z, function='cubic', smooth=0.3)
            # grid_z = rbf_func(grid_x, grid_y)

            # normalization
            z_min, z_max = np.nanmin(grid_z), np.nanmax(grid_z)
            if z_max > z_min:
                grid_z = (grid_z - z_min) / (z_max - z_min)
            else:
                grid_z = np.zeros_like(grid_z)



            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='Spectral', edgecolor='k', linewidth=0.3, alpha=0.95, vmin=0, vmax=0.8)

            # grid
            # ax.plot_wireframe(grid_x, grid_y, grid_z, color='black', linewidth=0.2, alpha=0.5)

            cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.02)
            # cbar.set_label('Normalized Performance', fontsize=18)

            if self.system_name == 'h2':

                cbar.set_label('Normalized Throughput', fontsize=26)
            else:

                cbar.set_label('Normalized Runtime', fontsize=26)


            # for global_opt_index in global_opt_indices:
            #     ax.scatter(x[global_opt_index], y[global_opt_index], grid_z.flatten()[global_opt_index],
            #                color='black', marker='*', s=200, edgecolors='white', linewidths=1.5,
            #                label='Global Optimum' if global_opt_index == global_opt_indices[0] else "")

            ax.set_xlabel('#D1', fontsize=26, labelpad=15)
            ax.set_ylabel('#D2', fontsize=26, labelpad=15)
            # ax.set_zlabel('Normalized Performance', fontsize=16)

            ax.view_init(elev=35, azim=220)

            file_path = os.path.join(self.full_landscape_path, f'{workload}_3D_normalized.pdf')
            plt.savefig(file_path, bbox_inches='tight', format='pdf')
            plt.close()

    def performance_distribution_visualization(self, config_2d_dict):
        for workload, config_2d in config_2d_dict.items():
            performance = self.perf_dict[workload]

            plt.figure(figsize=(10, 6))
            sns.set_style("whitegrid")
            sns.kdeplot(performance, fill=True, color='skyblue', linewidth=2, bw_adjust=1.0)
            plt.title(f'Performance KDE Distribution for {workload} - {self.system_name}', fontsize=16)
            plt.xlabel('Performance Value', fontsize=14)
            plt.ylabel('Density', fontsize=14)

            file_path = os.path.join(self.perf_distribution_path, f'{workload}_performance_distribution_kde.pdf')
            plt.savefig(file_path, bbox_inches='tight', format='pdf')
            # plt.show()
            plt.close()

    def top_index_overlap_3d_visualization(self, config_2d_dict, top_percent=0.1):
        workloads = list(config_2d_dict.keys())
        num_workloads = len(workloads)
        overlap_results = {}

        for i in range(num_workloads):
            for j in range(i + 1, num_workloads):
                wl1, wl2 = workloads[i], workloads[j]

                config = config_2d_dict[wl1]
                perf1 = self.perf_dict[wl1]
                perf2 = self.perf_dict[wl2]

                if self.system_name in ['h2', 'mysql', 'postgresql', 'httpd', 'tomcat', 'lighttpd']:
                    # top 5% index
                    k1 = max(1, int(len(perf1) * top_percent))
                    k2 = max(1, int(len(perf2) * top_percent))
                    top_idx1 = set(np.argsort(-perf1)[:k1])
                    top_idx2 = set(np.argsort(-perf2)[:k2])
                    overlap_idx = top_idx1 & top_idx2
                elif self.system_name in ['batik', 'dconvert', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3', 'x265']:
                    k1 = max(1, int(len(perf1) * top_percent))
                    k2 = max(1, int(len(perf2) * top_percent))
                    top_idx1 = set(np.argsort(perf1)[:k1])
                    top_idx2 = set(np.argsort(perf2)[:k2])
                    overlap_idx = top_idx1 & top_idx2

                overlap_rate = len(overlap_idx) / min(len(top_idx1), len(top_idx2))
                overlap_results[(wl1, wl2)] = overlap_rate

                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection='3d')

                config1_top = config[list(top_idx1)]
                config2_top = config[list(top_idx2)]
                perf1_top = perf1[list(top_idx1)]
                perf2_top = perf2[list(top_idx2)]

                def plot_circles(ax, coords, perf, edge_color, face_color):
                    for x, y, z in zip(coords[:, 0], coords[:, 1], perf):
                        ax.scatter([x], [y], [z], s=80,
                                   facecolors=face_color,
                                   edgecolors=edge_color,
                                   linewidths=1.5,
                                   marker='o')

                plot_circles(ax, config1_top, perf1_top, edge_color='#006400', face_color='#b6e2b6')  # 深绿 + 淡绿
                plot_circles(ax, config2_top, perf2_top, edge_color='#8B0000', face_color='#f4cccc')  # 深红 + 淡红

                for idx in overlap_idx:
                    x, y = config[idx]
                    z1 = perf1[idx]
                    z2 = perf2[idx]
                    ax.plot([x, x], [y, y], [z1, z2], color='gray', linestyle='--', linewidth=1.5)

                ax.set_xlabel("#D1", fontsize=16, labelpad=15)
                ax.set_ylabel("#D2", fontsize=16, labelpad=15)
                if self.system_name in ['mysql', 'postgresql', 'httpd', 'tomcat', 'h2']:
                    ax.set_zlabel("Throughput", fontsize=16)
                else:
                    ax.set_zlabel("Runtime", fontsize=16)

                ax.tick_params(axis='both', labelsize=12)
                # ax.tick_params(axis='z', labelsize=12)
                # ax.view_init(elev=25, azim=135)
                # ax.grid(False)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('w')
                ax.yaxis.pane.set_edgecolor('w')
                ax.zaxis.pane.set_edgecolor('w')
                ax.set_title(f"Overlap Rate: {overlap_rate:.2f}", fontsize=14)

                # label1_desc = self.parse_workload_label(wl1)
                # label2_desc = self.parse_workload_label(wl2)
                label1_desc = wl1
                label2_desc = wl2


                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label=f'{label1_desc}',
                           markerfacecolor='#b6e2b6', markeredgecolor='#006400', markersize=8),
                    Line2D([0], [0], marker='o', color='w', label=f'{label2_desc}',
                           markerfacecolor='#f4cccc', markeredgecolor='#8B0000', markersize=8)
                    # Line2D([0], [0], linestyle='--', color='gray', label='Overlap link')
                ]
                ax.legend(handles=legend_elements, fontsize=10, loc='best')

                output_name = f"{wl1}_vs_{wl2}_top_overlap_3D.pdf"
                file_path = os.path.join(self.top_5_percent, output_name)
                
                plt.savefig(file_path, bbox_inches='tight', pad_inches=0.4, format='pdf')
                plt.close()
                print(f"✅: {output_name}（overlap: {overlap_rate:.2f}）")

        heatmap_matrix = np.zeros((num_workloads, num_workloads))
        for i in range(num_workloads):
            for j in range(num_workloads):
                if i == j:
                    heatmap_matrix[i, j] = 1.0
                elif (workloads[i], workloads[j]) in overlap_results:
                    heatmap_matrix[i, j] = overlap_results[(workloads[i], workloads[j])]
                elif (workloads[j], workloads[i]) in overlap_results:
                    heatmap_matrix[i, j] = overlap_results[(workloads[j], workloads[i])]

        plt.figure(figsize=(10, 8))
        im = plt.imshow(heatmap_matrix, cmap='coolwarm', vmin=0, vmax=1)

        plt.xticks(ticks=np.arange(num_workloads), labels=[wl for wl in workloads],
                   rotation=45, ha='right', fontsize=10)
        plt.yticks(ticks=np.arange(num_workloads), labels=[wl for wl in workloads],
                   fontsize=10)

        for i in range(num_workloads):
            for j in range(num_workloads):
                plt.text(j, i, f"{heatmap_matrix[i, j]:.2f}", ha='center', va='center', color='black',
                         fontsize=9)

        plt.title(f"Top {top_percent*100}% Overlap Rate Heatmap", fontsize=14)
        plt.colorbar(im, shrink=0.8, label="Overlap Rate")
        plt.tight_layout()

        heatmap_path = os.path.join(self.local_optima_overlap_heatmap_path, "top_overlap_heatmap.pdf")
        plt.savefig(heatmap_path, bbox_inches='tight', format='pdf')

        plt.close()
        print(f"✅: top_overlap_heatmap.pdf")

        return overlap_results

    def top_index_overlap_3d_visualization2(self, config_2d_dict, top_percent=0.1, threshold=0.1):
        workloads = list(config_2d_dict.keys())
        num_workloads = len(workloads)
        overlap_results = {}

        for i in range(num_workloads):
            for j in range(i + 1, num_workloads):
                wl1, wl2 = workloads[i], workloads[j]
                config = config_2d_dict[wl1]
                perf1 = self.perf_dict[wl1]
                perf2 = self.perf_dict[wl2]

                if self.system_name in ['h2', 'mysql', 'postgresql', 'httpd', 'tomcat']:
                    k1 = max(1, int(len(perf1) * top_percent))
                    k2 = max(1, int(len(perf2) * top_percent))
                    top_idx1 = set(np.argsort(-perf1)[:k1])
                    top_idx2 = set(np.argsort(-perf2)[:k2])
                else:
                    k1 = max(1, int(len(perf1) * top_percent))
                    k2 = max(1, int(len(perf2) * top_percent))
                    top_idx1 = set(np.argsort(perf1)[:k1])
                    top_idx2 = set(np.argsort(perf2)[:k2])

                overlap_idx = top_idx1 & top_idx2
                overlap_rate = len(overlap_idx) / min(len(top_idx1), len(top_idx2))
                overlap_results[(wl1, wl2)] = overlap_rate

        mat = np.zeros((num_workloads, num_workloads))
        for i in range(num_workloads):
            for j in range(num_workloads):
                if i == j:
                    mat[i, j] = 1.0
                elif (workloads[i], workloads[j]) in overlap_results:
                    mat[i, j] = overlap_results[(workloads[i], workloads[j])]
                elif (workloads[j], workloads[i]) in overlap_results:
                    mat[i, j] = overlap_results[(workloads[j], workloads[i])]

        df = pd.DataFrame(mat, index=workloads, columns=workloads)

        fig, ax = plt.subplots(figsize=(10, 11))
        cmap = cm.get_cmap("RdBu")
        norm = mcolors.Normalize(vmin=0, vmax=1)

        for i in range(num_workloads):
            for j in range(0, i + 1):
                val = df.iloc[i, j]
                if val < threshold:
                    continue

                radius = 0.35 #0.45 * np.sqrt(val)
                color = cmap(norm(val))
                circle = patches.Circle((j + 0.5, i + 0.5), radius=radius,
                                        facecolor=color, alpha=0.75, edgecolor='black', linewidth=0.5)
                ax.add_patch(circle)

                ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha='center', va='center',
                        fontsize=12, color='black')

        ax.set_xlim(0, num_workloads)
        ax.set_ylim(0, num_workloads)
        ax.set_xticks(np.arange(num_workloads) + 0.5)
        ax.set_yticks(np.arange(num_workloads) + 0.5)
        # ax.set_xticklabels(workloads, rotation=45, ha='right', fontsize=9)
        # ax.set_yticklabels(workloads, fontsize=9)
        ax.invert_yaxis()
        # ax.xaxis.tick_top()

        short_labels = [f"W{i + 1}" for i in range(num_workloads)]
        ax.set_xticks(np.arange(num_workloads) + 0.5)
        ax.set_yticks(np.arange(num_workloads) + 0.5)
        ax.set_xticklabels(short_labels, ha='right', fontsize=9)
        ax.set_yticklabels(short_labels, fontsize=9)

        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

        print("\n📘 Workload mapping：")
        for i, wl in enumerate(workloads):
            print(f"  W{i + 1}: {wl}")


        for i in range(num_workloads):
            for j in range(0, i + 1):
                ax.add_patch(patches.Rectangle((j, i), 1, 1, fill=False,
                                               edgecolor='gray', linewidth=0.5))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        norm = plt.Normalize(0, 1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.06, pad=0.05)
        cbar.set_label("Overlap Rate", fontsize=14)

        # ax.set_title(f"{self.system_name}".upper(), fontsize=13, pad=14)
        plt.tight_layout()

        output_path = os.path.join(self.local_optima_overlap_heatmap_path, "top_overlap_lower_triangle_circles.pdf")
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()

        print(f"✅: top_overlap_lower_triangle_circles.pdf")

        return overlap_results

    def parse_workload_label(self, raw_label):

        result = []

        time_match = re.search(r'(\d+)s', raw_label)
        if time_match:
            result.append(f"time: {time_match.group(1)}")

        tbls_match = re.search(r'(\d+)tbls', raw_label)
        if tbls_match:
            result.append(f"tables: {tbls_match.group(1)}")
        size_match = re.search(r'(\d+)size', raw_label)
        if size_match:
            result.append(f"table-size: {size_match.group(1)}")

        result.append(f"threads: 4")
        return '[' + '; '.join(result) + ']'

