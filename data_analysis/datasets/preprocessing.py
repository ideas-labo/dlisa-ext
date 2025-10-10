import os
import pandas as pd

# 自定义系统名称列表
system_list = ["x265"]  # 举例，根据实际替换

# 数据集目录
base_dir = "../datasets"

# 遍历系统
for system in system_list:
    system_path = os.path.join(base_dir, system)
    if not os.path.isdir(system_path):
        print(f"System path not found: {system_path}")
        continue

    for file_name in os.listdir(system_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(system_path, file_name)
            try:
                df = pd.read_csv(file_path)
                # 删除最后两列
                df = df.iloc[:, :-5]
                # 覆盖保存
                df.to_csv(file_path, index=False)
                print(f"Processed: {file_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

def clean_csv_folder(folder_path, output_folder=None):
    # 获取所有 csv 文件
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    file_paths = [os.path.join(folder_path, f) for f in csv_files]

    # 读取所有文件为 DataFrame
    dataframes = [pd.read_csv(fp) for fp in file_paths]

    # 找出所有包含 'error' 的行索引（只看最后一列）
    invalid_indices = set()
    for df in dataframes:
        perf_col = df.columns[-1]
        invalid_rows = df[perf_col].astype(str).str.lower() == 'error'
        invalid_indices.update(df[invalid_rows].index.tolist())

    print(f"Found {len(invalid_indices)} invalid rows due to 'error' values.")

    # 删除所有文件中这些行
    cleaned_dfs = [df.drop(index=invalid_indices).reset_index(drop=True) for df in dataframes]

    # 输出清洗后的文件
    output_folder = output_folder or os.path.join(folder_path)
    os.makedirs(output_folder, exist_ok=True)

    for fname, df in zip(csv_files, cleaned_dfs):
        output_path = os.path.join(output_folder, fname)
        df.to_csv(output_path, index=False)

    print(f"Cleaned CSV files saved to: {output_folder}")

# 示例调用（替换成你真实的路径）
clean_csv_folder("../datasets/x265")
