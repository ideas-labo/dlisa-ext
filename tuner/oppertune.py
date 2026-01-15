
import os
import math
import random
from dataclasses import dataclass
from typing import List, Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from utils.logger import Logger
from utils.config_manager import ConfigManager


@dataclass
class OPPerTuneParams:
    # per-workload evaluation budget
    max_measurements: int = 90

    # EXP3 / Exponential-Weights hyperparams
    delta: float = 0.10          # exploration mixing (0~1); higher => more uniform exploration
    eta: float = 0.20            # learning rate for exponential update

    # numerical stability
    reward_clip: float = 50.0    # clip reward into [-reward_clip, reward_clip] before update

    # logging/output
    results_root: str = "results"
    algo_name: str = "OPPerTune"
    save_per_workload_csv: bool = True
    save_aggregate_csv: bool = True


class ExponentialWeightsSlatesPolicy:
    """
    Slates version (recommended): one categorical bandit per dimension.
    Action space = Cartesian product of value_domains, but sampling is factorized across dimensions.

    Sampling:
      p_hat_i = (1 - delta)*p_i + delta*(1/|D_i|)
      x_i ~ Categorical(p_hat_i)

    Update (importance-weighted exponential update):
      p_i[x_i] *= exp( eta * reward / p_hat_i[x_i] )
      normalize p_i
    """
    def __init__(self, value_domains: List[List[Any]], delta: float, eta: float, rng: np.random.RandomState):
        self.value_domains = value_domains
        self.d = len(value_domains)
        self.delta = float(delta)
        self.eta = float(eta)
        self.rng = rng

        # For each dim i: p_i is a vector over |D_i|
        self.p_list: List[np.ndarray] = []
        for dom in self.value_domains:
            k = len(dom)
            if k <= 0:
                raise ValueError("Empty value domain detected.")
            self.p_list.append(np.ones(k, dtype=float) / k)

        # Keep last sampling info for update
        self._last_choice_indices: Optional[List[int]] = None
        self._last_p_hat_list: Optional[List[np.ndarray]] = None

    def sample(self) -> List[Any]:
        choice_indices: List[int] = []
        p_hat_list: List[np.ndarray] = []

        for i, p in enumerate(self.p_list):
            k = p.shape[0]
            # exploration mixing
            p_hat = (1.0 - self.delta) * p + (self.delta / k)
            # make sure it's a valid distribution
            p_hat = np.clip(p_hat, 1e-12, 1.0)
            p_hat = p_hat / p_hat.sum()

            idx = int(self.rng.choice(k, p=p_hat))
            choice_indices.append(idx)
            p_hat_list.append(p_hat)

        self._last_choice_indices = choice_indices
        self._last_p_hat_list = p_hat_list

        # decode indices -> actual values
        config = [self.value_domains[i][choice_indices[i]] for i in range(self.d)]
        return config

    def update(self, reward: float) -> None:
        if self._last_choice_indices is None or self._last_p_hat_list is None:
            raise RuntimeError("Call sample() before update().")

        for i in range(self.d):
            idx = self._last_choice_indices[i]
            p_hat = self._last_p_hat_list[i]
            denom = float(p_hat[idx])

            # importance-weighted exponential update (EXP3-style)
            # p[idx] *= exp(eta * reward / denom)
            mult = math.exp(self.eta * (reward / max(denom, 1e-12)))

            p = self.p_list[i]
            p[idx] *= mult
            # normalize
            s = p.sum()
            if not np.isfinite(s) or s <= 0:
                # reset to uniform if numerical issues happen
                k = p.shape[0]
                p[:] = 1.0 / k
            else:
                p[:] = p / s

    def update_with_config(self, config: List[Any], reward: float) -> None:
        """
        Update policy using a GIVEN config (used for warm-start seeds),
        without relying on the internal state set by sample().
        This matches the same EXP3-slates update rule used in update().
        """
        if len(config) != self.d:
            raise ValueError(f"Config dimension mismatch: expected {self.d}, got {len(config)}")

        for i in range(self.d):
            dom = self.value_domains[i]
            try:
                idx = dom.index(config[i])  # categorical value -> index
            except ValueError:
                # If a value is not in domain (shouldn't happen), skip this dimension
                continue

            p = self.p_list[i]
            k = p.shape[0]

            # same mixing as in sample()
            p_hat = (1.0 - self.delta) * p + (self.delta / k)
            p_hat = np.clip(p_hat, 1e-12, 1.0)
            p_hat = p_hat / p_hat.sum()

            denom = float(p_hat[idx])
            mult = math.exp(self.eta * (reward / max(denom, 1e-12)))

            p[idx] *= mult
            s = p.sum()
            if not np.isfinite(s) or s <= 0:
                p[:] = 1.0 / k
            else:
                p[:] = p / s



class OPPerTuneTuner:
    """
    Minimal OPPerTune (no-context) tuner for your framework:
    - Workloads are processed sequentially (job-style)
    - Under each workload, run max_measurements online updates
    - Policy is preserved across workloads (adaptation by online learning state inheritance)
    - Evaluation is done via ConfigManager.evaluate(..., fallback='worst') (you said always worst)
    """
    def __init__(
        self,
        system: str,
        workloads: List[str],
        run_id: int,
        optimization_goal: str,
        max_measurements: int,
        fallback: str = "worst",
        params: Optional[OPPerTuneParams] = None,
        seed: int = 0,
        initial_seeds=None,
        common_seeds=None,

    ):
        self.initial_seeds = initial_seeds or []
        self.common_seeds = common_seeds or []
        self.system = system
        self.workloads = workloads
        self.run_id = run_id
        self.optimization_goal = optimization_goal  # "minimum" or "maximum"
        self.max_measurements = int(max_measurements)
        self.fallback = fallback  # you said always "worst"

        self.params = params or OPPerTuneParams(max_measurements=max_measurements)
        # Override (in case user passes max_measurements separately)
        self.params.max_measurements = self.max_measurements

        self.cm = ConfigManager(system, optimization_goal)

        # RNG for reproducibility
        self.rng = np.random.RandomState(seed + 10007 * (run_id + 1))

        # policy will be created after we load first workload (to ensure value_domains ready)
        self.policy: Optional[ExponentialWeightsSlatesPolicy] = None


        # aggregate records (across workloads)
        self.all_records: List[Dict[str, Any]] = []

    def _to_reward(self, perf: float) -> float:
        # Make reward "higher is better"
        if self.optimization_goal == "maximum":
            r = float(perf)
        elif self.optimization_goal == "minimum":
            r = -float(perf)
        else:
            raise ValueError(f"Unknown optimization_goal: {self.optimization_goal}")

        # Clip for numerical stability (important for exp updates)
        if self.params.reward_clip is not None:
            r = float(np.clip(r, -self.params.reward_clip, self.params.reward_clip))
        return r

    def _get_config_columns(self) -> List[str]:
        # Use dataset columns if possible (exclude last perf column)
        if self.cm.data_df is not None:
            cols = list(self.cm.data_df.columns)
            if len(cols) >= 2:
                return cols[:-1]
        # fallback generic names
        return [f"x{i}" for i in range(self.cm.config_dim)]

    def _init_policy_if_needed(self):
        if self.policy is None:
            # value_domains is loaded in ConfigManager.__init__ via init_config_space()
            self.policy = ExponentialWeightsSlatesPolicy(
                value_domains=self.cm.value_domains,
                delta=self.params.delta,
                eta=self.params.eta,
                rng=self.rng,
            )

    def run(self):
        for workload_id, workload in enumerate(self.workloads):
            print(f"\n[OpperTune | RUN {self.run_id}] >>> Workload {workload_id + 1}/{len(self.workloads)}: {workload}")

            # Switch workload (policy is preserved across workloads)
            self.cm.set_workload(workload)
            self._init_policy_if_needed()

            header = self.cm.get_config_header()
            logger = Logger(
                base_dir=f"results/run{self.run_id}",
                algorithm="OpperTune",
                system=f"{self.system}",
                workload=f"{workload}",
                header=header
            )

            all_results = []  # List[(config, perf)]
            evaluated = set()  # per-workload evaluated configs (tuples)

            # Upper bound to avoid infinite loops when space is small / policy collapses
            max_resample_attempts = 200

            # ------------------------------
            # (A) Warm-start with shared seeds (counted into the same budget)
            # ------------------------------
            # NOTE: please ensure OPPerTuneTuner.__init__ stores these:
            #   self.initial_seeds: List[List]
            #   self.common_seeds:  List[List]
            warmstart_pool = []
            if hasattr(self, "initial_seeds") and self.initial_seeds:
                warmstart_pool.extend(self.initial_seeds)
            if hasattr(self, "common_seeds") and self.common_seeds:
                warmstart_pool.extend(self.common_seeds)

            # Deduplicate warm-start candidates while preserving order
            seen_ws = set()
            ws_unique = []
            for cfg in warmstart_pool:
                key = tuple(cfg)
                if key not in seen_ws:
                    seen_ws.add(key)
                    ws_unique.append(cfg)

            # Evaluate warm-start configs and update policy
            budget_used = 0
            for cfg in ws_unique:
                if budget_used >= self.max_measurements:
                    break

                key = tuple(cfg)
                if key in evaluated:
                    continue

                (cfg_eval, perf) = self.cm.evaluate([cfg], fallback=self.fallback)[0]
                evaluated.add(tuple(cfg_eval))
                all_results.append((cfg_eval, perf))

                reward = self._to_reward(perf)
                self.policy.update_with_config(cfg_eval, reward)

                budget_used += 1

            remaining_budget = self.max_measurements - budget_used
            if remaining_budget <= 0:
                logger.log_all_results(all_results)
                continue

            # ------------------------------
            # (B) Online tuning with EXP3-slates for the remaining budget
            # ------------------------------
            for step in range(1, remaining_budget + 1):
                attempts = 0
                while True:
                    cfg = self.policy.sample()
                    key = tuple(cfg)

                    if key not in evaluated:
                        break

                    attempts += 1
                    if attempts >= max_resample_attempts:
                        print(
                            f"[WARN][OpperTune] Early stop at step={step} (remaining budget) for workload={workload}: "
                            f"cannot sample a new config after {max_resample_attempts} attempts."
                        )
                        cfg = None
                        break

                if cfg is None:
                    break

                (cfg_eval, perf) = self.cm.evaluate([cfg], fallback=self.fallback)[0]
                evaluated.add(tuple(cfg_eval))
                all_results.append((cfg_eval, perf))

                reward = self._to_reward(perf)
                self.policy.update(reward)

            logger.log_all_results(all_results)









