import os
import pandas as pd
import re


class Utility:
    @staticmethod
    def extract_parts(file_name):
        """Extract the prefix letters and trailing number from a file name like 'workload12.csv'."""
        match = re.match(r'([a-zA-Z]+)(\d+)', file_name)
        if match:
            return match.group(1), int(match.group(2))
        return file_name, 0

    @staticmethod
    def get_valid_csv_files(system_path):
        """Return sorted list of valid CSV filenames excluding '_2d' and '_FDC'."""
        files = [
            f for f in os.listdir(system_path)
            if f.endswith('.csv') and '_2d' not in f and '_FDC' not in f
        ]
        return sorted(files, key=Utility.extract_parts)

    @staticmethod
    def load_perf_data(system_path, system_name):
        """Load the 'throughput' column as performance data for all workloads."""
        performance_dict = {}
        for file_name in Utility.get_valid_csv_files(system_path):
            file_path = os.path.join(system_path, file_name)
            df = pd.read_csv(file_path)
            workload_name = file_name.split('.csv')[0]
            if system_name == 'mysql' or system_name == 'postgresql' or system_name == 'mysql_sampling' or system_name == 'h2':
                if 'throughput' in df.columns:
                    performance_dict[workload_name] = df['throughput']
                else:
                    raise ValueError(f"'throughput' column not found in {file_name}")
            elif system_name == 'gcc' or system_name == 'clang':
                if 'run_time' in df.columns:
                    performance_dict[workload_name] = df['run_time']
                else:
                    raise ValueError(f"'throughput' column not found in {file_name}")
            elif system_name == 'httpd' or system_name == 'tomcat':
                if 'RPS' in df.columns:
                    performance_dict[workload_name] = df['RPS']
                else:
                    raise ValueError(f"'throughput' column not found in {file_name}")
            elif system_name == 'lighttpd':
                performance = pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna()
                performance_dict[workload_name] = performance.values
            else:
                performance_dict[workload_name] = df['time']

        return performance_dict

    @staticmethod
    def load_config_data(system_path, system_name):
        """Load the first 20 columns as configuration data for all workloads."""
        config_dict = {}
        for file_name in Utility.get_valid_csv_files(system_path):
            file_path = os.path.join(system_path, file_name)
            df = pd.read_csv(file_path)
            workload_name = file_name.split('.csv')[0]
            if system_name == 'mysql' or system_name == 'postgresql' or system_name == 'httpd' or system_name == 'tomcat':
                config_dict[workload_name] = df.iloc[:, :10]
            elif system_name == 'gcc' or system_name == 'clang':
                config_dict[workload_name] = df.iloc[:, :10]
            else:
                config_dict[workload_name] = df.iloc[:, :-1]

        return config_dict

    @staticmethod
    def load_config_and_perf_data(system_path, system_name):
        """Load configuration and performance data for all workloads, ensuring alignment."""
        config_dict = {}
        performance_dict = {}

        for file_name in Utility.get_valid_csv_files(system_path):
            file_path = os.path.join(system_path, file_name)
            df = pd.read_csv(file_path)
            workload_name = file_name.split('.csv')[0]

            # Identify performance column based on system
            if system_name in ['mysql', 'postgresql', 'mysql_sampling', 'h2']:
                perf_col = 'throughput'
            elif system_name in ['gcc', 'clang']:
                perf_col = 'run_time'
            elif system_name in ['httpd', 'tomcat']:
                perf_col = 'RPS'
            elif system_name == 'lighttpd':
                perf_col = df.columns[-1]
            elif system_name == 'x265':
                perf_col = 'EncodeTime'
            else:
                perf_col = df.columns[-1]

            # Convert to numeric, drop invalid rows
            df[perf_col] = pd.to_numeric(df[perf_col], errors='coerce')
            valid_rows = df[perf_col].notna()

            # Store aligned config and perf
            config_dict[workload_name] = df.loc[valid_rows, :].iloc[:, :-1]  # all but last col
            performance_dict[workload_name] = df.loc[valid_rows, perf_col].values  # as array

        return config_dict, performance_dict
