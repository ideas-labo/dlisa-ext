import os
import pandas as pd

# 自定义系统名称列表
system_list = ["tomcat"]  # 举例，根据实际替换

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
                df = df.iloc[:, :-2]
                # 覆盖保存
                df.to_csv(file_path, index=False)
                print(f"Processed: {file_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
