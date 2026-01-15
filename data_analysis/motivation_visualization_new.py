import os
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
from matplotlib.font_manager import FontProperties


def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    markers = ['o', '^', 's', 'H', 'H']  # 不同的标记
    colors = ['#FBC64F', '#62ACFF', '#91BF6F', '#FF7E9C', '#FE85AB',
              '#A0A0A0']

    # 选择要分析的系统和环境
    selected_system = 'xz'
    # selected_environments = ['sac_5', 'sac_6', 'sac_7', 'sac_8', 'sac_9']  # 例如: ['SAC6', 'SAC7']
    # selected_environments = ['spear_0', 'spear_1', 'spear_2', 'spear_3']
    # selected_environments = ['x264_0', 'x264_4', 'x264_7', 'x264_8']
    # selected_environments = ['corona','cranium','flame', 'france'] # batlik
    # selected_environments = ['jpeg-large', 'jpeg-medium', 'jpeg-small', 'png-large', 'png-medium', 'png-small']# dconvert
    # selected_environments = ['voter-2', 'voter-16', 'smallbank-1', 'smallbank-10']#h2
    # selected_environments = ['beethoven.wav', 'dual-channel.wav', 'helix.wav', 'single-channel.wav']#jump3r
    # selected_environments = ['enwik8', 'misc', 'large', 'vmlinux']  # kanzi
    # selected_environments = ['ambivert.wav.tar', 'artificl.tar', 'deepfield.tar', 'enwik8.tar', 'fannie_mae_500k.tar'] #lrzip
    # selected_environments = ['Netflix_Crosswalk_4096x2160_60fps_10bit_420_short.y4m', 'pedestrian_area_1080p25_short.y4m', 'blue_sky_1080p25_short.y4m', 'sd_city_4cif_short.y4m']#x264
    selected_environments = ['artificl.tar', 'large.tar', 'ambivert.tar', 'uiq2-4.bin'] #lrzip #xz
    # selected_environments = ['AUFNIRA_z3.637557.smt2', 'LRA_formula_277.smt2', 'QF_AUFBV_891_sqlite3.smt2', 'QF_BV_bench_3176.smt2']


    performance_data = {}
    configuration_data = {}

    for system in systems:
        if system == selected_system:
            system_folder = 'datasets/' + system
            environment_csv_files = sorted([f for f in os.listdir(system_folder) if
                                            os.path.splitext(f)[0] in selected_environments])
            # environment_csv_files = ['smallbank-10.csv', 'smallbank-1.csv', 'voter-2.csv', 'voter-16.csv']
            # environment_csv_files = ['deepfield.csv', 'large.csv', 'vmlinux.csv', 'misc.csv']  # enwik8
            # environment_csv_files = ['jpeg-large.csv', 'jpeg-medium.csv', 'jpeg-small.csv', 'svg-large.csv']
            environment_csv_files = ['misc.tar.csv', 'large.tar.csv', 'uiq2-4.bin.csv', 'enwik8.tar.csv'] #lrzip #xz

            for i, environment_csv_file in enumerate(environment_csv_files):
                environment_name = os.path.splitext(environment_csv_file)[0]
                data = pd.read_csv(os.path.join(system_folder, environment_csv_file))

                column_names = data.columns.tolist()

                performance_data[environment_name] = data[column_names[-1]]
                configuration_data[environment_name] = data[column_names[0:-1]]

            # local_optima_indices = compute_local_optima_via_clustering(configuration_data, performance_data, n_clusters=50)
            local_optima_indices = compute_top_k_local_optima(performance_data, top_k=50)

            # there are some local optima that remain local optima in multiple environments. The probability of these solutions being local optima when transferred to a new environment is higher.
            plot_5(performance_data, local_optima_indices, colors, markers, system)

            # plot_4(performance_data, local_optima_indices, colors, system)


            # plot_2(performance_data, local_optima_indices, colors, system, selected_environments)


            # plot_3(performance_data, local_optima_indices, colors, system, selected_environments)



            plt.figure(figsize=(12, 8))

            target_environment = selected_environments[-1]
            target_optima_indices = local_optima_indices[target_environment]
            target_performance = performance_data[target_environment]
            target_local_optima = target_performance[target_optima_indices]

            plt.scatter(target_performance.index, target_performance, marker='.',
                        color=colors[len(selected_environments) - 1],
                        label='performance of target: ' + target_environment)

            plt.scatter(target_optima_indices, target_local_optima, marker='*',
                        color='black', alpha=0.5, label='local optima of target: ' + target_environment)

            for i, environment in enumerate(selected_environments[:-1]):

                optima_indices = local_optima_indices[environment]
                optimal_performances = target_performance[optima_indices]
                plt.scatter(optima_indices, optimal_performances, marker=markers[i], color=colors[i],
                            label='local optima of ' + environment)

            plt.legend()
            plt.show()


def plot_5(performance_data, local_optima_indices, colors, markers, system):

    fig = plt.figure(figsize=(7, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    # workload_to_int = {'smallbank-10': 1, 'smallbank-1': 2,  'voter-2': 3, 'voter-16': 4}
    # workload_to_int = {'deepfield': 1, 'misc': 4, 'large': 2, 'vmlinux': 3}
    # workload_to_int = {'jpeg-large': 3, 'jpeg-medium': 2, 'jpeg-small': 1, 'svg-large': 4}
    workload_to_int = {'misc.tar': 1, 'large.tar': 2, 'enwik8.tar': 4, 'uiq2-4.bin': 3} #lrzip #xz



    common_optima_indices = set.intersection(*[set(optima) for optima in local_optima_indices.values()])

    # exclude some environments to see the connection lines more clearly

    # keys_to_consider = ['large', 'misc', 'vmlinux']
    # keys_to_consider = ['smallbank-10', 'smallbank-1',  'voter-2', 'voter-16']
    # keys_to_consider = ['jpeg-large', 'jpeg-medium']
    keys_to_consider = ['large.tar', 'enwik8.tar', 'uiq2-4.bin']
    values_to_intersect = [set(local_optima_indices[key]) for key in keys_to_consider]
    common_optima_indices = set.intersection(*values_to_intersect)


    for i, (environment, optima_indices) in enumerate(local_optima_indices.items()):
        performance = performance_data[environment]
        optimal_performances = performance[optima_indices]


        optima_data = {'Configuration Ids': optima_indices, 'Performance': optimal_performances}
        optima_df = pd.DataFrame(optima_data)

        optima_df_sorted = optima_df.sort_values(by='Performance', ascending=True)

        csv_filename = f'{system}_{environment}.csv'
        optima_df_sorted.to_csv(csv_filename, index=False)

        # 绘制局部最优点
        ax.scatter(optima_indices, [workload_to_int[environment]] * len(optima_indices), optimal_performances,
                   color=colors[i % len(colors)],
                   marker=markers[i % len(markers)], label=f'{environment}', edgecolors='black', alpha=0.6,
                   s=50)

    for optima_index in common_optima_indices:
        y_values = []
        z_values = []
        for env in local_optima_indices:
            if optima_index in local_optima_indices[env]:
                y_values.append(workload_to_int[env])
                z_values.append(performance_data[env][optima_index])

        if len(y_values) >= 1:
            ax.plot(
                [optima_index] * len(y_values),
                y_values,
                z_values,
                color='#B1AAAA',
                linestyle='--',
                marker='.',  # Add marker
                markevery=1  # Show marker at every point
            )

    ax.set_yticks(list(workload_to_int.values()))
    ax.set_yticklabels(list(workload_to_int.keys()))

    ax.set_xlabel('#C', fontname='Times New Roman', fontsize=26, labelpad=35)
    ax.set_ylabel('#W', fontname='Times New Roman', fontsize=26, labelpad=35)
    if system == 'h2':
        ax.set_zlabel('Throughput', fontname='Times New Roman', fontsize=26, labelpad=20)
    else:
        ax.set_zlabel('Runtime', fontname='Times New Roman', fontsize=26, labelpad=20)

    yticklabels = ax.get_yticklabels()

    for label in yticklabels:
        label.set_rotation(0)

    ax.set_yticklabels([label.get_text() for label in yticklabels])

    font_prop = FontProperties(family='Times New Roman', size=20)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, columnspacing=-0.1, handletextpad=-0.5,
    #           labelspacing=0.01, prop=font_prop)

    # legend = ax.legend(
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, -0.05),
    #     ncol=4,  # ✅ 4 列
    #     columnspacing=-0.3,  # 列间距
    #     handletextpad=-0.5,  # marker 与文字间距
    #     labelspacing=0.1,  # 行间距（多行时）
    #     prop=font_prop,
    #     frameon=False,  # ✅ 去边框
    # )
    # ax.set_xlim(0, 5000)

    ax.tick_params(axis='y', which='major', labelsize=13, pad=13)
    ax.tick_params(axis='z', which='major', labelsize=13, pad=13)
    ax.tick_params(axis='x', which='major', labelsize=13, pad=13)
    ax.view_init(elev=35, azim=-45)
    # plt.show()

    plt.savefig(f'{system}_new.pdf', format='pdf', bbox_inches='tight',)

    # 你原本画图完之后
    tikz = generate_tikz_for_plot5(
        performance_data=performance_data,
        local_optima_indices=local_optima_indices,
        colors=colors,
        markers=markers,
        system=system,
        workload_to_int=workload_to_int,
        keys_to_consider=keys_to_consider,  # 跟你现在一致；想全交集就传 None
        view_elev=35,
        view_azim=-45,
    )

    save_tikz(tikz, f"{system}_plot5_optima.tex")


def _sanitize_tex(s: str) -> str:
    # workload 名字一般安全，这里做最基本处理
    return s.replace('_', r'\_')

def _mpl_marker_to_pgf(mark: str) -> str:
    # 你现在用的 markers 里常见映射：o ^ s
    return {
        'o': '*',
        '^': 'triangle*',
        's': 'square*',
        'D': 'diamond*',
        'v': 'triangle*',
        '<': 'triangle*',
        '>': 'triangle*',
        'p': 'pentagon*',
        'h': 'hexagon*',
        'x': 'x',
        '+': '+',
        '.': '*',  # 你连线 marker='.'，pgfplots 用 * 近似即可
    }.get(mark, '*')

def _fmt_num(x) -> str:
    # pgfplots 对科学计数法也能处理；这里尽量稳定输出
    if x is None:
        return "nan"
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return "nan"
        return f"{x:.10g}"
    except Exception:
        return "nan"

def generate_tikz_for_plot5(
    performance_data: dict,
    local_optima_indices: dict,
    colors: list,
    markers: list,
    system: str,
    workload_to_int: dict = None,
    keys_to_consider: list = None,
    view_elev: int = 35,
    view_azim: int = -45,
    fig_width: str = "12cm",
    fig_height: str = "8cm",
    legend_cols: int = 2,
    line_color: str = "B1AAAA",
    line_style: str = "dashed",
    point_size: str = "2.2pt",
    opacity: float = 0.6,
) -> str:
    """
    把 plot_5 的“局部最优点 + common optima 虚线连接”导出为可编译的 TikZ/PGFPlots。
    - performance_data[workload] -> 1D array-like (len=N)
    - local_optima_indices[workload] -> list[int] (索引是 Configuration Id)
    - colors: 形如 ['#F0C36D', '#7DA6D9', ...] 或不带 '#'
    - markers: 形如 ['o','^','s',...]
    """

    if workload_to_int is None:
        # 默认沿用你当前脚本的 mapping
        workload_to_int = {'deepfield': 1, 'large': 2, 'vmlinux': 3, 'misc': 4}

    # 计算 common optima（与你现在写法一致：可选 keys_to_consider 做更清晰的连线）
    if keys_to_consider is None:
        values_to_intersect = [set(v) for v in local_optima_indices.values()]
    else:
        values_to_intersect = [set(local_optima_indices[k]) for k in keys_to_consider if k in local_optima_indices]
    common_optima_indices = set.intersection(*values_to_intersect) if values_to_intersect else set()

    # 组装工作负载顺序：按 y 值排序，保证 ytick label 顺序稳定
    workloads_sorted = sorted(workload_to_int.keys(), key=lambda k: workload_to_int[k])
    # 只保留你实际有数据的 workload
    workloads_sorted = [w for w in workloads_sorted if (w in performance_data and w in local_optima_indices)]

    # 颜色定义
    def_color_lines = []
    plot_styles = []  # 每个 workload 的 addplot 样式 + 数据 table
    legend_entries = []

    for i, w in enumerate(workloads_sorted):
        c = colors[i % len(colors)]
        c = c.lstrip('#')
        color_name = f"wlcolor{i+1}"
        def_color_lines.append(rf"\definecolor{{{color_name}}}{{HTML}}{{{c}}}")

        pgf_mark = _mpl_marker_to_pgf(markers[i % len(markers)])

        # 取出局部最优点
        opt_idx = list(local_optima_indices[w])
        perf = np.asarray(performance_data[w])
        y = workload_to_int[w]

        rows = []
        for idx in opt_idx:
            # x=Configuration Id(idx), y=workload_to_int, z=performance
            z = perf[idx]
            rows.append(f"{idx} {y} {_fmt_num(z)}")

        # pgfplots table：每行 "x y z \\"
        table_body = " \\\\\n".join(rows)

        style = (
            r"only marks,"
            rf"mark={pgf_mark},"
            rf"mark size={point_size},"
            rf"draw=black,"
            rf"fill={color_name},"
            rf"fill opacity={opacity},"
            rf"draw opacity=1"
        )

        plot_styles.append(
            rf"""\addplot3+[{style}]
table[row sep=\\] {{
x y z \\
{table_body} \\
}};"""
        )
        legend_entries.append(rf"\addlegendentry{{{_sanitize_tex(w)}}}")

    # 连线（每个 common optima index 一条线，连接所有出现该 index 的 workload）
    conn_lines = []
    for optima_index in sorted(common_optima_indices):
        coords = []
        for w in workloads_sorted:
            if optima_index in set(local_optima_indices[w]):
                y = workload_to_int[w]
                z = np.asarray(performance_data[w])[optima_index]
                coords.append(f"({optima_index},{y},{_fmt_num(z)})")
        if len(coords) >= 1:
            conn_lines.append(
                rf"\addplot3+[color=connline,{line_style}] coordinates {{"
                + " ".join(coords)
                + r"};"
            )

    # 轴标签与你 plot_5 对齐
    zlabel = "Throughput" if system == "h2" else "Runtime"

    # ytick/label
    ytick_vals = [workload_to_int[w] for w in workloads_sorted]
    ytick_labels = [ _sanitize_tex(w) for w in workloads_sorted ]

    # 视角：pgfplots 用 view={azim}{elev}
    tikz = rf"""
% =========================
% Auto-generated by generate_tikz_for_plot5()
% Compile with: pdflatex
% Requires: \usepackage{{pgfplots}} \pgfplotsset{{compat=1.18}}
% =========================
\begin{{tikzpicture}}
\pgfplotsset{{compat=1.18}}
\definecolor{{connline}}{{HTML}}{{{line_color.lstrip('#')}}}
{chr(10).join(def_color_lines)}

\begin{{axis}}[
  width={fig_width},
  height={fig_height},
  view={{{view_azim}}}{{{view_elev}}},
  grid=both,
  xlabel={{Configuration Id}},
  ylabel={{Workload}},
  zlabel={{{zlabel}}},
  ytick={{{",".join(str(v) for v in ytick_vals)}}},
  yticklabels={{{",".join("{" + lab + "}" for lab in ytick_labels)}}},
  tick label style={{font=\small}},
  label style={{font=\large}},
  legend style={{
    at={{(0.5,-0.12)}},
    anchor=north,
    legend columns={legend_cols},
    /tikz/every even column/.append style={{column sep=6pt}}
  }},
]
{chr(10).join(plot_styles)}

% connection lines for common optima
{chr(10).join(conn_lines)}

{chr(10).join(legend_entries)}
\end{{axis}}
\end{{tikzpicture}}
""".strip()

    return tikz


def save_tikz(tex_str: str, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex_str + "\n")



def plot_1(performance_data, local_optima_indices, colors, system):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')


    common_optima_indices = set.intersection(*[set(optima) for optima in local_optima_indices.values()])


    for i, (environment, optima_indices) in enumerate(local_optima_indices.items()):
        performance = performance_data[environment]
        optimal_performances = performance[optima_indices]


        ax.scatter(optima_indices, [i] * len(optima_indices), optimal_performances, color=colors[i % len(colors)],
                   label=environment)


        for optima_index in optima_indices:
            if optima_index in common_optima_indices:

                ax.plot([optima_index] * len(local_optima_indices), list(range(len(local_optima_indices))),
                        [performance_data[env][optima_index] for env in local_optima_indices], color='#B1AAAA',
                        linestyle='--')
            else:
                pass


    ax.set_xlabel('Configuration Ids', fontname='Times New Roman')
    ax.set_ylabel('Workload', fontname='Times New Roman')
    # ax.set_zscale('log')
    # ax.set_zlim(0,15.5)
    ax.set_zlabel('Execution Time', fontname='Times New Roman')
    ax.legend()
    plt.title("DCONVERT", fontname='Times New Roman')

    plt.show()
    plt.savefig('DCONVERT.pdf')


def plot_4(performance_data, local_optima_indices, colors, system):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    environments = list(local_optima_indices.keys())[:-1]
    common_optima_indices = set.intersection(*[set(local_optima_indices[env]) for env in environments])

    for i, environment in enumerate(environments):
        optima_indices = local_optima_indices[environment]
        performance = performance_data[environment]
        optimal_performances = performance[optima_indices]

        ax.scatter(optima_indices, [i] * len(optima_indices), optimal_performances, color=colors[i % len(colors)],
                   label=environment)

        for optima_index in common_optima_indices:
            ax.plot([optima_index] * len(environments),
                    list(range(len(environments))),
                    [performance_data[env][optima_index] for env in environments],
                    color='black', linestyle='--')

    ax.set_xlabel('Indices')
    ax.set_ylabel('Environment')
    ax.set_zlabel('Performance')
    ax.legend()
    plt.title("System: " + system)

    plt.show()



def plot_2(performance_data, local_optima_indices, colors, system, selected_environments):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    target_environment = selected_environments[-1]
    target_optima_indices = local_optima_indices[target_environment]
    target_performance = performance_data[target_environment]
    target_local_optima = target_performance[target_optima_indices]

    for i, environment in enumerate(selected_environments[:-1]):
        optima_indices = local_optima_indices[environment]
        performance = performance_data[environment]

        ax.scatter(optima_indices, [i] * len(optima_indices), performance[optima_indices],
                   label=environment, color=colors[i % len(colors)])

        for optima_index in optima_indices:
            if optima_index in target_optima_indices:
                target_performance_value = target_performance[optima_index]

                ax.plot([optima_index, optima_index], [i, len(selected_environments) - 1],
                        [performance[optima_index], target_performance_value],
                        color=colors[i % len(colors)], linestyle='--')

    ax.scatter(target_optima_indices, [len(selected_environments) - 1] * len(target_optima_indices),
               target_local_optima,
               label=target_environment, color=colors[len(selected_environments) - 1])

    ax.set_xlabel('Indices')
    ax.set_ylabel('Environment')
    ax.set_zlabel('Performance')

    ax.legend()
    plt.title("System:" + system)
    plt.show()

def plot_3(performance_data, local_optima_indices, colors, system, selected_environments):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    target_environment = selected_environments[-1]
    target_optima_indices = local_optima_indices[target_environment]
    target_performance = performance_data[target_environment]
    target_local_optima = target_performance[target_optima_indices]

    for i, current_environment in enumerate(selected_environments[:-1]):
        current_optima_indices = local_optima_indices[current_environment]
        current_performance = performance_data[current_environment]

        ax.scatter(current_optima_indices, [i] * len(current_optima_indices),
                   current_performance[current_optima_indices],
                   label=current_environment, color=colors[i % len(colors)])

        unique_optima_indices = []
        for idx in current_optima_indices:
            if idx in target_optima_indices:
                is_unique = True
                for other_environment in selected_environments[:-1]:
                    if other_environment != current_environment and idx in local_optima_indices[
                        other_environment]:
                        is_unique = False
                        break
                if is_unique:
                    unique_optima_indices.append(idx)

        for optima_index in unique_optima_indices:
            target_index = optima_index
            target_performance_value = target_local_optima[target_index]
            ax.plot([optima_index, optima_index], [i, len(selected_environments) - 1],
                    [current_performance[optima_index], target_performance_value],
                    color=colors[i % len(colors)], linestyle='--')

    ax.scatter(target_optima_indices, [len(selected_environments) - 1] * len(target_optima_indices),
               target_local_optima,
               label=target_environment, color=colors[len(selected_environments) - 1])

    ax.set_xlabel('Indices')
    ax.set_ylabel('Environment')
    ax.set_zlabel('Performance')
    plt.title("System:" + system)

    ax.legend()
    plt.show()


def compute_local_optima_via_clustering(configuration_data, performance_data, n_clusters=100):
    local_optima_indices = {}


    representative_configs = next(iter(configuration_data.values()))


    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='hamming', linkage='average')
    labels = clustering.fit_predict(representative_configs)


    for environment, configs in configuration_data.items():

        environment_performance = performance_data[environment]
        local_optima_indices[environment] = []

        for cluster_label in np.unique(labels):
            cluster_indices = np.where(labels == cluster_label)[0]
            best_performance_index = cluster_indices[np.argmin(environment_performance[cluster_indices])]
            local_optima_indices[environment].append(best_performance_index)

    return local_optima_indices


def compute_top_k_local_optima(performance_data, top_k=50):
    local_optima_indices = {}

    for environment, performances in performance_data.items():

        best_indices = np.argsort(performances)[:top_k]
        # best_indices = np.argsort(performances)[::-1][:top_k] # for maximization problems
        local_optima_indices[environment] = best_indices.tolist()

    return local_optima_indices


if __name__ == "__main__":
    main()






