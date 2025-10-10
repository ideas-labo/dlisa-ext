import random

from utils.evolutionary_planning import EvolutionaryPlanner
from utils.config_manager import ConfigManager
from utils.logger import Logger


class DSOGATuner:
    def __init__(self, system, workloads, run_id, optimization_goal, initial_seeds, common_seeds, max_generation,
                 pop_size, fallback=None):

        self.system = system
        self.workloads = workloads
        self.run_id = run_id
        self.optimization_goal = optimization_goal
        self.initial_seeds = initial_seeds
        self.common_seeds = common_seeds
        self.fallback = fallback

        self.max_generation = max_generation
        self.pop_size = pop_size
        self.prev_optimized_pop = None

    def run(self):
        config_manager = ConfigManager(system=self.system, optimization_goal=self.optimization_goal)

        for workload_id, workload_name in enumerate(self.workloads):
            print(f"\n[DSOGA | RUN {self.run_id}] >>> Workload {workload_id + 1}/{len(self.workloads)}: {workload_name}")
            config_manager.set_workload(workload_name)

            header = config_manager.get_config_header()
            logger = Logger(
                base_dir=f"results/run{self.run_id}",
                algorithm="DSOGA",
                system=f"{self.system}",
                workload=f"{workload_name}",
                header=header
            )

            if workload_id == 0:
                if self.initial_seeds:
                    init_population = self.initial_seeds
                else:
                    init_population = config_manager.random_initialize_population(self.pop_size)
            else:
                reuse_count = int(self.pop_size * 0.8)
                random_count = self.pop_size - reuse_count

                reused = random.sample([cfg for cfg, _ in self.prev_optimized_pop], reuse_count)
                random_new = config_manager.random_initialize_population(random_count)

                init_population = reused + random_new


            planner = EvolutionaryPlanner(
                config_manager=config_manager,
                init_population=init_population,
                max_generation=self.max_generation,
                mutation_rate=0.1,
                crossover_rate=0.9,
                common_seeds=self.common_seeds,
                pop_size=self.pop_size,
                logger=logger,
                fallback=self.fallback
            )
            optimized_pop = planner.evolutionary_search_config()
            self.prev_optimized_pop = optimized_pop








