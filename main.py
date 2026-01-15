import os
import json
import time
import random

from tuner.femosaa import FEMOSAATuner
from tuner.seedea import SEEDEATuner
from tuner.dsoga import DSOGATuner
from tuner.dlisa import DLiSATuner
from tuner.lidos import LiDOSTuner
from tuner.dlisab import DLiSABTuner
from utils.config_manager import ConfigManager

from tuner.dlisab0 import DLiSAB0Tuner
from tuner.dlisab1 import DLiSAB1Tuner
from tuner.dlisab2 import DLiSAB2Tuner
from tuner.dlisab3 import DLiSAB3Tuner
from tuner.dlisab4 import DLiSAB4Tuner
from tuner.dlisab6 import DLiSAB6Tuner
from tuner.dlisab7 import DLiSAB7Tuner
from tuner.dlisab8 import DLiSAB8Tuner
from tuner.dlisab9 import DLiSAB9Tuner
from tuner.dlisabI import DLiSABITuner
from tuner.dlisabII import DLiSABIITuner
from tuner.dlisac import DLiSACTuner
from tuner.dlisabr1 import DLiSABR1Tuner
from tuner.dlisabr2 import DLiSABR2Tuner
from tuner.oppertune import OPPerTuneTuner


def main():
    runs = 100
    max_generation = 2
    pop_size = 20
    systems = ['batik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    # Algorithms for comparison
    # compared_algorithms = ['LiDOS', 'FEMOSAA', 'SEED-EA', 'D-SOGA', 'DLiSA', 'DLiSAB4'
    # BR1: max_generation=1; BR2: max_generation=2
    compared_algorithms = ['OpperTune']
    # Sensitivity analysis
    # compared_algorithms = ['DLiSAB1', 'DLiSAB2', 'DLiSAB3', 'DLiSAB4', 'DLiSAB6', 'DLiSAB7', 'DLiSAB8', 'DLiSAB9']
    # Ablation study
    # compared_algorithms = ['DLiSA-B4', 'DLiSA-BI', 'DLiSA-BII']

    # Fallback strategy for configurations not found in the dataset:
    # - 'worst': assign the worst observed performance value in the dataset (✅used in the paper)
    # - 'nearest': assign the performance of the most similar configuration (based on Hamming distance) in the dataset
    fallback = 'worst'

    start_time = time.time()

    for i in range(runs):
        print(f"\n========================Run {i + 1}/{runs}==========================")
        for system in systems:
            optimization_goal = 'maximum' if system in ['h2', 'mysql', 'postgresql', 'tomcat', 'httpd', 'lighttpd'] else 'minimum'
            config_manager = ConfigManager(system, optimization_goal)

            workload_order_file = f'order_files/order_{system}_{i}.json'
            if not os.path.exists(workload_order_file):
                workloads = sorted([f[:-4] for f in os.listdir(f'datasets/{system}') if f.endswith('.csv')])
                random.shuffle(workloads)
                os.makedirs(os.path.dirname(workload_order_file), exist_ok=True)
                with open(workload_order_file, 'w') as f:
                    json.dump(workloads, f)
            else:
                with open(workload_order_file, 'r') as f:
                    workloads = json.load(f)

            initial_seeds = config_manager.random_initialize_population(pop_size)  # ensuring initial pop for all algorithms are the same
            common_seeds = config_manager.random_initialize_common_seeds(pop_size // 2)
            for algo_name in compared_algorithms:
                print(f"\n[INFO] Running {algo_name} on {system}, run {i}")

                if algo_name == 'FEMOSAA':
                    tuner = FEMOSAATuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'SEED-EA':
                    tuner = SEEDEATuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'D-SOGA':
                    tuner = DSOGATuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'LiDOS':
                    tuner = LiDOSTuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSA':
                    tuner = DLiSATuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSA-B':
                    tuner = DLiSABTuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSAB0':
                    tuner = DLiSAB0Tuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSAB1':
                    tuner = DLiSAB1Tuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSAB2':
                    tuner = DLiSAB2Tuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSAB3':
                    tuner = DLiSAB3Tuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSAB4':
                    tuner = DLiSAB4Tuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSAB6':
                    tuner = DLiSAB6Tuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSAB7':
                    tuner = DLiSAB7Tuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSAB8':
                    tuner = DLiSAB8Tuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSAB9':
                    tuner = DLiSAB9Tuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSA-BI':
                    tuner = DLiSABITuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSA-BII':
                    tuner = DLiSABIITuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSA-C':
                    tuner = DLiSACTuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSABR1':
                    tuner = DLiSABR1Tuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'DLiSABR2':
                    tuner = DLiSABR2Tuner(system, workloads, i, optimization_goal, initial_seeds, common_seeds, max_generation, pop_size, fallback)
                elif algo_name == 'OpperTune':
                    tuner = OPPerTuneTuner(
                        system=system,
                        workloads=workloads,
                        run_id=i,
                        optimization_goal=optimization_goal,
                        max_measurements=90,
                        fallback=fallback,
                        seed=42,
                        initial_seeds=initial_seeds,
                        common_seeds=common_seeds,
                    )
                else:
                    print(f"[ERROR] Unknown algorithm: {algo_name}")
                    continue

                tuner.run()

    print(f"\nTotal time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
