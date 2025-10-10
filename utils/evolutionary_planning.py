import numpy as np
import random

from typing import List, Tuple
from copy import deepcopy
from scipy.spatial.distance import cdist


class EvolutionaryPlanner:
    def __init__(self, config_manager, init_population=None, max_generation=3,
                 mutation_rate=0.1, crossover_rate=0.9, common_seeds=None, pop_size=20, logger=None, env_selection_strategy=None, fallback=None, evaluated_common_seeds=None):
        self.config_manager = config_manager
        self.init_population = init_population
        self.max_generation = max_generation
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.common_seeds = common_seeds
        self.evaluated_configs = {}  # config tuple -> perf
        self.all_results = []
        self.pop_size = pop_size
        self.logger = logger
        self.env_selection_strategy = env_selection_strategy
        self.fallback = fallback  # Fallback strategy for configurations not found in the dataset
        self.evaluated_common_seeds = evaluated_common_seeds

    def evolutionary_search_config(self) -> List[Tuple[List[float], float]]:
        generation = 0
        population = deepcopy(self.init_population)
        if len(population) < self.pop_size:
            extra = self.config_manager.random_initialize_population(self.pop_size - len(population))
            population.extend(extra)

        if self.evaluated_common_seeds is None:
            population_pool = population + self.common_seeds
            evaluated_pop = self.config_manager.evaluate(population_pool, self.fallback)
        else:
            evaluated_pop = self.config_manager.evaluate(population, self.fallback)
            evaluated_pop = evaluated_pop + self.evaluated_common_seeds

        for cfg, perf in evaluated_pop:
            self.evaluated_configs[tuple(cfg)] = perf
            self.all_results.append((cfg, perf))
        parents = self.environment_selection(evaluated_pop, self.pop_size, self.config_manager.optimization_goal)
        self.logger.log_generation(generation, parents)

        while generation < self.max_generation:
            print(f"[INFO] Generation {generation + 1}/{self.max_generation}")

            offspring = []
            seen = set(tuple(cfg) for cfg, _ in self.all_results)
            while len(offspring) < self.pop_size:
                p1, p2 = self.tournament_selection(parents, self.config_manager.optimization_goal)
                c1, c2 = self.single_point_crossover(p1, p2)
                c1 = self.mutation(c1)
                c2 = self.mutation(c2)

                for child in [c1, c2]:
                    t = tuple(child)
                    if t not in self.evaluated_configs and t not in seen:
                        offspring.append(child)
                        seen.add(t)
                    if len(offspring) >= self.pop_size:
                        break

            evaluated_offspring = self.config_manager.evaluate(offspring, self.fallback)
            for cfg, perf in evaluated_offspring:
                self.evaluated_configs[tuple(cfg)] = perf
                self.all_results.append((cfg, perf))

            combined = parents + evaluated_offspring
            parents = self.environment_selection(combined, self.pop_size, self.config_manager.optimization_goal)
            generation += 1
            self.logger.log_generation(generation, parents)
        self.logger.log_all_results(self.all_results)

        return parents

    def tournament_selection(self, parents, optimization_goal):
        ind1a, ind1b = random.sample(parents, 2)
        ind2a, ind2b = random.sample(parents, 2)

        if optimization_goal == "maximum":
            p1 = ind1a if ind1a[1] > ind1b[1] else ind1b
            p2 = ind2a if ind2a[1] > ind2b[1] else ind2b
        else:  # assume "minimum"
            p1 = ind1a if ind1a[1] < ind1b[1] else ind1b
            p2 = ind2a if ind2a[1] < ind2b[1] else ind2b

        return p1[0], p2[0]

    def uniform_crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)

        child1, child2 = [], []
        for v1, v2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child1.append(v1)
                child2.append(v2)
            else:
                child1.append(v2)
                child2.append(v1)
        return child1, child2

    def single_point_crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)

        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        return child1, child2

    def mutation(self, individual: List[float]) -> List[float]:
        mutated = individual[:]
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                candidates = self.config_manager.value_domains[i]
                original = mutated[i]
                alternatives = [v for v in candidates if v != original]
                if alternatives:
                    mutated[i] = random.choice(alternatives)
        return mutated

    def environment_selection(self, evaluated_configs, pop_size, optimization_goal):
        if self.env_selection_strategy is None:
            reverse = optimization_goal == "maximum"
            sorted_pop = sorted(evaluated_configs, key=lambda x: x[1], reverse=reverse)
            return sorted_pop[:pop_size]
        elif self.env_selection_strategy == "LiDOS":

            configs = [cfg for cfg, _ in evaluated_configs]
            performances = np.array([perf for _, perf in evaluated_configs])
            filtered_configs = self.filter_numeric_dims(configs)

            fa = []
            for i in range(len(filtered_configs)):
                xi = filtered_configs[i]
                distances = cdist([xi], filtered_configs)[0]
                distances[i] = np.inf
                neighbors_idx = np.argsort(distances)[:5]  # 5 nearest neighbors

                max_diff_perf = max(abs(performances[idx] - performances[i]) for idx in neighbors_idx)
                fa.append(max_diff_perf)
            fa = np.array(fa)


            w = 1.0
            g1 = performances + w * fa
            g2 = performances - w * fa
            if self.config_manager.optimization_goal == "maximum":
                g1, g2 = -g1, -g2

            objectives = list(zip(g1, g2))

            fronts = self.fast_non_dominated_sort(objectives)

            selected = []
            for front in fronts:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                    if len(selected) == pop_size:
                        break
                else:
                    crowding_distances = self.crowding_distance_assignment(objectives, front)
                    sorted_front = sorted(zip(front, crowding_distances), key=lambda x: -x[1])
                    selected.extend([idx for idx, _ in sorted_front[:pop_size - len(selected)]])
                    break

            return [evaluated_configs[idx] for idx in selected]

    def dominates(self, obj1, obj2):
        return all(o <= p for o, p in zip(obj1, obj2)) and any(o < p for o, p in zip(obj1, obj2))

    def fast_non_dominated_sort(self, objectives):

        S = [[] for _ in range(len(objectives))]
        n = [0 for _ in range(len(objectives))]
        rank = [0 for _ in range(len(objectives))]
        front = [[]]
        for p in range(len(objectives)):
            for q in range(len(objectives)):
                if self.dominates(objectives[p], objectives[q]):
                    S[p].append(q)
                elif self.dominates(objectives[q], objectives[p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                front[0].append(p)
        i = 0
        while front[i]:
            next_front = []
            for p in front[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            front.append(next_front)
            i += 1
        return front[:-1]  # Last front will be empty

    def crowding_distance_assignment(self, objectives, front):
        num_objects = len(objectives[0])
        distances = [0] * len(front)

        if len(front) <= 1:
            return [float('inf')] * len(front)

        for m in range(num_objects):
            sorted_front = sorted(front, key=lambda i: objectives[i][m])
            distances[front.index(sorted_front[0])] = float('inf')
            distances[front.index(sorted_front[-1])] = float('inf')

            min_obj = objectives[sorted_front[0]][m]
            max_obj = objectives[sorted_front[-1]][m]

            if max_obj == min_obj:
                continue

            for i in range(1, len(sorted_front) - 1):
                distances[front.index(sorted_front[i])] += (objectives[sorted_front[i + 1]][m] -
                                                            objectives[sorted_front[i - 1]][m]) / (
                                                                   max_obj - min_obj)

        return distances

    def filter_numeric_dims(self, configs: List[List]) -> List[List[float]]:

        arr = np.array(configs, dtype=object)
        num_rows, num_cols = arr.shape

        numeric_col_mask = []
        for col in range(num_cols):
            col_vals = arr[:, col]
            if all(isinstance(v, (int, float, np.integer, np.floating)) for v in col_vals):
                numeric_col_mask.append(True)
            else:
                numeric_col_mask.append(False)

        numeric_arr = arr[:, numeric_col_mask]
        return numeric_arr.tolist()



