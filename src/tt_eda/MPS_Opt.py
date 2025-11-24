    
import numpy as np
from typing import List
import time
from MPS_TN import MPS
from typing import List, Dict
import copy

class Opt_MPS():
    def __init__(self,
                 bbox_func,
                 MPS_instance: MPS,
                 num_bbox_func_query: int = int(5.E+3),
                 tot_num_samples: int = 100,
                 num_top_samples: int = 10,
                 learning_rate: float = 0.01,
                 num_sweeps: int = 1,
                 num_grad_steps: int = 2,
                 stagnation_mode: bool = False,
                 stagnation_patience: int = 10,
                 base_mutation_rate: float = 0.0,            
                 stagnation_active_mutation_rate: float = 0.05,
                 perturbation_strength_on_stagnation: float = 0.05,
                 ):
        
        if bbox_func is None:
            raise ValueError("bbox_func must be provided.")
        if MPS_instance is None:
            raise ValueError("MPS_instance must be provided.")

        self.bbox_func = bbox_func
        self.MPS = MPS_instance
        self.num_bbox_func_query = num_bbox_func_query
        self.tot_num_samples = tot_num_samples
        self.num_top_samples = num_top_samples
        self.learning_rate = learning_rate
        self.num_sweeps = num_sweeps
        self.num_grad_steps = num_grad_steps
        self.stagnation_mode = stagnation_mode
        self.stagnation_patience = stagnation_patience
        self.base_mutation_rate = base_mutation_rate
        self.stagnation_active_mutation_rate = stagnation_active_mutation_rate
        self.perturbation_strength_on_stagnation = perturbation_strength_on_stagnation
        self.last_y_min_improvement_iter = 0
        self.in_stagnation_exploration_mode = False

    def bbox_func_to_opt(self, config: np.ndarray) -> float:
        return self.bbox_func(config)

    def bbox_func_eval_samples(self, samples: np.ndarray) -> np.ndarray:
        try:
            return np.apply_along_axis(self.bbox_func_to_opt, axis=1, arr=samples)
        except Exception as e:
            print(f"Error during batch evaluation of bbox_func: {e}")
            return np.full(samples.shape[0], np.nan)

    def log_likelihood(self, config: np.ndarray) -> float:
        try:
            cond_probs = self.MPS.calc_1Norm_conditional_probs(config)
            return np.sum(np.log(np.maximum(cond_probs, 1e-20)))
        except Exception as e:
            print(f"Error calculating log likelihood for config {config}: {e}")
            return -np.inf

    def log_likelihood_samples(self, samples: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self.log_likelihood, axis=1, arr=samples)

    def mutate_samples(self, samples: np.ndarray, mutation_rate: float = 0.0, rng: np.random.RandomState = None) -> np.ndarray:
        """
        Flip a site to a different symbol with prob=mutation_rate.
        Uses MPS phys dim from first node.
        """

        # Learnt it from https://arxiv.org/abs/2412.19780, cool stuff
        if mutation_rate <= 0.0:
            return samples

        if rng is None:
            rng = np.random

        d_values = int(self.MPS.nodes_ls[0].leg_dims[1])
        mutated = samples.copy()
        for r in range(mutated.shape[0]):
            mask = rng.random(mutated.shape[1]) < mutation_rate
            if np.any(mask):
                for c in np.where(mask)[0]:
                    old = mutated[r, c]
                    new = rng.randint(0, d_values - 1)   

                    if new >= old:
                        new += 1
                    mutated[r, c] = new
        return mutated

    def run(self, base_seed=1, use_stagnation=False):
        if use_stagnation is None:
            use_stagnation = self.stagnation_mode
        if use_stagnation:
            # return self.opt_engine_weighted_stag(base_seed=base_seed)
            # return self.opt_engine_annealing(base_seed=base_seed)
            return self.opt_engine_weighted_stag(base_seed=base_seed)
        else:
            return self.opt_engine_born(base_seed=base_seed)


    def opt_engine(self, base_seed=None):

        if base_seed is None:
            base_seed = int.from_bytes(np.random.bytes(4), byteorder='big')
        rng = np.random.RandomState(base_seed)

        y_min = np.inf
        x_min = None
        history = []
        updates_skipped = 0
        start_time = time.time()

        print(f"\n--- Running Optimization ---")
        print(f"    Budget M={self.num_bbox_func_query}, K={self.tot_num_samples}, k={self.num_top_samples}")
        print(f"    Update params: lr={self.learning_rate}, sweeps={self.num_sweeps}, mut={self.base_mutation_rate}")

        total_evals = 0
        iteration = 0

        while total_evals < self.num_bbox_func_query:
            iteration += 1
            iter_seed = rng.randint(0, 2**32 - 1)

            samples_to_take = min(self.tot_num_samples, self.num_bbox_func_query - total_evals)
            if samples_to_take <= self.num_top_samples:
                print(f"    Warning: Remaining budget ({self.num_bbox_func_query - total_evals}) <= k ({self.num_top_samples}). Stopping early.")
                break
            
            current_samples_np = self.MPS.seq_sample_from(num_samples=samples_to_take, seed=iter_seed)
            current_samples_np = self.mutate_samples(current_samples_np, mutation_rate=self.base_mutation_rate, rng=rng)
        
            
            total_evals += samples_to_take
            sample_values_np = np.asarray(self.bbox_func_eval_samples(current_samples_np), dtype=float).reshape(-1)
            finite_mask = np.isfinite(sample_values_np)


            finite_mask = np.isfinite(sample_values_np)                         # shape (B,)
            samples_to_consider = current_samples_np[finite_mask, :]            # filter rows
            values_to_consider = sample_values_np[finite_mask] 


            ##### COLLAPSE CHECK ######

            # I want to do a check here to see if the MPS has collapsed to a single configuration, because in that case the sampling will produce identical samples and mode collapse takes place
            # Especially this causes NLL to approach identially zero, and the optimization gets stuck, no exploration nothing!!

            ###### COLLAPSE CHECK ENDS ######


            if not np.any(finite_mask):
                last_y = history[-1][1] if history else np.nan
                history.append((total_evals, last_y))
                updates_skipped += 1
                continue

            samples_to_consider = current_samples_np[finite_mask]
            values_to_consider = sample_values_np[finite_mask]
            current_k_effective = min(self.num_top_samples, samples_to_consider.shape[0])

            if current_k_effective == 0:
                last_y = history[-1][1] if history else np.nan
                history.append((total_evals, last_y))
                updates_skipped += 1
                continue

            best_indices = np.argsort(values_to_consider)[:current_k_effective]
            best_samples_for_update = samples_to_consider[best_indices]

            current_iter_min_val = values_to_consider[best_indices[0]]

            if current_iter_min_val < y_min:
                y_min = current_iter_min_val
                x_min = samples_to_consider[best_indices[0]]

            history.append((total_evals, y_min if np.isfinite(y_min) else np.nan))


            self.MPS.one_site_batch_grad_default(
                    samples=best_samples_for_update,
                    num_sweeps=self.num_sweeps,
                    learning_rate=self.learning_rate
                )


            if iteration % 5 == 0 or iteration == 1:
                # print(best_samples_for_update)
                # loglikes= self.MPS.log_likelihood_1norm_samples(best_samples_for_update)
                # print('loglikes',loglikes)
                loss = -np.mean(self.MPS.log_likelihood_1norm_samples(best_samples_for_update))
                y_min_print = y_min if np.isfinite(y_min) else float('nan')
                print(f"    Iter {iteration} [Evals {total_evals}/{self.num_bbox_func_query}]: Best y_min = {y_min_print:.3e}  LL = {loss:.3e}")
            
        # print(best_samples_for_update)
        end_time = time.time()
        print(f"--- Finished Optimization Engine ({(end_time - start_time):.2f} sec) ---")
        print(f"    Best value y_min: {y_min}")
        print(f"    Best configuration x_min: {x_min}")
        return x_min, y_min, history

    def opt_engine_stag(self, base_seed=None):
        
        if base_seed is None:
            base_seed = int.from_bytes(np.random.bytes(4), byteorder='big')
        rng = np.random.RandomState(base_seed)

        # mps_unif = copy.deepcopy(self.MPS)

        y_min = np.inf
        x_min = None
        history = []
        updates_skipped = 0
        start_time = time.time()

        print(f"\n--- Running Optimization (stagnation-aware) ---")
        print(f"    Budget M={self.num_bbox_func_query}, K={self.tot_num_samples}, k={self.num_top_samples}")
        print(f"    Params: lr={self.learning_rate}, sweeps={self.num_sweeps}")
        print(f"    Stagnation: patience={self.stagnation_patience}, mut_active={self.stagnation_active_mutation_rate}, perturb={self.perturbation_strength_on_stagnation}")

        total_evals = 0
        iteration = 0
        self.last_y_min_improvement_iter = 0
        self.in_stagnation_exploration_mode = False

        while total_evals < self.num_bbox_func_query:
            iteration += 1
            iter_seed = rng.randint(0, 2**32 - 1)

            if (not self.in_stagnation_exploration_mode) and (y_min != np.inf) and (iteration - self.last_y_min_improvement_iter > self.stagnation_patience):
                print(f"    Iter {iteration}: stagnation detected. Enabling exploration.")
                self.in_stagnation_exploration_mode = True
                # self.MPS.gaussian_perturbation(sigma=self.perturbation_strength_on_stagnation)
                        # self.MPS.uniform_perturbation(sigma=self.perturbation_strength_on_stagnation)
                self.MPS.mix_with_uniform(alpha=self.perturbation_strength_on_stagnation)
                print(f"Applied MPS.uniform_perturbation(sigma={self.perturbation_strength_on_stagnation}).")
                self.last_y_min_improvement_iter = iteration

            exploring = self.in_stagnation_exploration_mode
            mut_rate = self.stagnation_active_mutation_rate if exploring else self.base_mutation_rate
            samples_to_take = min(self.tot_num_samples+50, self.num_bbox_func_query - total_evals) if exploring else min(self.tot_num_samples, self.num_bbox_func_query - total_evals)

            if samples_to_take <= self.num_top_samples:
                if total_evals < self.num_bbox_func_query and samples_to_take > 0:
                    current_samples_np = self.MPS.seq_sample_from(num_samples=samples_to_take, seed=iter_seed)
                    current_samples_np = self.mutate_samples(current_samples_np, mutation_rate=mut_rate, rng=rng)
                    vals = self.bbox_func_eval_samples(current_samples_np)
                    total_evals += samples_to_take
                    finite = np.isfinite(vals)
                    if np.any(finite):
                        m = np.argmin(vals[finite])
                        if vals[finite][m] < y_min:
                            y_min = vals[finite][m]
                            x_min = current_samples_np[finite][m]
                        history.append((total_evals, y_min))
                print("    Stopping: not enough budget for a full iteration.")
                break

            current_samples_np = self.MPS.seq_sample_from(num_samples=samples_to_take, seed=iter_seed)
            current_samples_np = self.mutate_samples(current_samples_np, mutation_rate=mut_rate, rng=rng)
            vals = self.bbox_func_eval_samples(current_samples_np)
            total_evals += samples_to_take

            finite = np.isfinite(vals)
            if not np.any(finite):
                last_y = history[-1][1] if history else np.nan
                history.append((total_evals, last_y))
                updates_skipped += 1
                print(f"    Iter {iteration}: all samples non-finite; skipping update.")
                continue

            S = current_samples_np[finite]
            V = vals[finite]
            k_eff = min(self.num_top_samples, S.shape[0])
            best_idx = np.argsort(V)[:k_eff]
            elites = S[best_idx]
            iter_min = V[best_idx[0]]

            improved = False
            if iter_min < y_min:
                y_min = iter_min
                x_min = elites[0]
                self.last_y_min_improvement_iter = iteration
                improved = True
                if self.in_stagnation_exploration_mode:
                    print(f"    Iter {iteration}: improved to {y_min:.3e}; leaving exploration.")
                    self.in_stagnation_exploration_mode = False

            history.append((total_evals, y_min if np.isfinite(y_min) else np.nan))

            # GRADDD
            self.MPS.one_site_batch_grad_default(
                    samples=elites,
                    num_sweeps=self.num_sweeps,
                    learning_rate=self.learning_rate
                )

        
            # if iteration % 2 == 0:
            #     print(f"    Iter {iteration}: applying small uniform mix to avoid over-concentration.")
            #     self.MPS.mix_with_uniform(alpha=0.05)

            if iteration % 5 == 0 or iteration == 1 or improved:
                nll = -np.mean(self.MPS.log_likelihood_1norm_samples(elites)) 
                # self.MPS.mix_with_uniform(alpha=0.05)              
                print(f"    Iter {iteration} [Evals {total_evals}/{self.num_bbox_func_query}]: "
                      f"y_min={y_min:.3e}  NLL={nll:.4f}  mut={mut_rate:.3f}")
            
            if nll <= 5:  # distribution too peaked
                print('distribution too peaked')
           
            # if iteration % 5 == 0 or iteration == 1 or improved or self.in_stagnation_exploration_mode:
            #     nll = -np.mean(self.log_likelihood_samples(elites))                
            #     print(f"    Iter {iteration} [Evals {total_evals}/{self.num_bbox_func_query}]: "
            #           f"y_min={y_min:.3e}  NLL={nll:.4f}  mut={mut_rate:.3f}")
                
            # if nll <= 10:  # distribution too peaked
            #     # print(f"Iter {iteration}: NLL ≈ 0, applying perturbation.")
            #     self.MPS.mix_with_uniform(alpha=0.5)
            #     print(f"    Iter {iteration}: NLL={nll:.4f} <= 0.5; applied MPS.uniform_perturbation(sigma={self.perturbation_strength_on_stagnation}).")
            #     # print(f"    NLL={nll:.4f} <= 0.5; resetting MPS to uniform.")
            #     # self.MPS.nodes_ls = copy.deepcopy(mps_reset.nodes_ls)
            #     # self.num_top_samples = self.num_top_samples + 10  # temporarily increase k to diversify
            #     continue  # skip gradient update

        end = time.time()
        print(f"--- Finished (stagnation-aware) in {(end - start_time):.2f}s ---")
        print(f"    Best y_min: {y_min}")
        print(f"    Best x_min: {x_min.tolist() if x_min is not None else None}")
        return x_min, y_min, history

    def opt_engine_weighted_stag(self, base_seed=1, weighting="logrank", temperature=0.05):
        
        if base_seed is None:
            base_seed = int.from_bytes(np.random.bytes(4), byteorder='big')
        
        rng = np.random.RandomState(base_seed)

        y_min = np.inf
        x_min = None
        history = []
        updates_skipped = 0

        start_time = time.time()

        last_y_min_improvement_iter = 0
        in_stagnation_exploration_mode = False
        exploration_iters_left = 0

        print(f"\n--- Running Optimization (weighted, scheme={weighting}) ---")
        print(f"    Budget M={self.num_bbox_func_query}, K={self.tot_num_samples}, k={self.num_top_samples}")
        print(f"    Update params: lr={self.learning_rate}, sweeps={self.num_sweeps}, mut={self.base_mutation_rate}")
        if weighting in ("softmax", "exp"):
            print(f"    Weighting temperature: {temperature}")

        total_evals = 0
        iteration = 0

        while total_evals < self.num_bbox_func_query:
            iteration += 1
            iter_seed = rng.randint(0, 2**32 - 1)

            if (not in_stagnation_exploration_mode) and (y_min != np.inf) and (iteration - last_y_min_improvement_iter > self.stagnation_patience):
                print(f"    Iter {iteration}: stagnation detected. Enabling exploration (mixing with uniform, increasing mutation rate).")
                in_stagnation_exploration_mode = True
                exploration_iters_left = 5  
                if hasattr(self.MPS, 'mix_with_uniform'):
                    try:
                        self.MPS.mix_with_uniform(alpha=self.perturbation_strength_on_stagnation)
                        print(f"    Applied MPS.mix_with_uniform(alpha={self.perturbation_strength_on_stagnation}).")
                    except Exception as e_perturb:
                        print(f"    ERROR during MPS mix_with_uniform: {e_perturb}")
                else:
                    print("    Warning: MPS has no 'mix_with_uniform'; skipping.")
                last_y_min_improvement_iter = iteration

            if in_stagnation_exploration_mode and exploration_iters_left > 0:
                mut_rate = self.stagnation_active_mutation_rate
                exploration_iters_left -= 1
                if exploration_iters_left == 0:
                    in_stagnation_exploration_mode = False
            else:
                mut_rate = self.base_mutation_rate

            samples_to_take = min(self.tot_num_samples, self.num_bbox_func_query - total_evals)
            if samples_to_take <= self.num_top_samples:
                print(f"    Warning: Remaining budget ({self.num_bbox_func_query - total_evals}) <= k ({self.num_top_samples}). Stopping early.")
                break


            current_samples_np = self.MPS.seq_sample_from(num_samples=samples_to_take, seed=iter_seed)
            current_samples_np = self.mutate_samples(current_samples_np, mutation_rate=mut_rate, rng=rng)
            total_evals += samples_to_take
            sample_values_np = np.asarray(self.bbox_func_eval_samples(current_samples_np), dtype=float).reshape(-1)
            finite_mask = np.isfinite(sample_values_np)                         
            samples_to_consider = current_samples_np[finite_mask, :]            
            values_to_consider = sample_values_np[finite_mask] 

            if not np.any(finite_mask):
                last_y = history[-1][1] if history else np.nan
                history.append((total_evals, last_y))
                updates_skipped += 1
                continue

            samples_to_consider = current_samples_np[finite_mask]
            values_to_consider = sample_values_np[finite_mask]
            current_k_effective = min(self.num_top_samples, samples_to_consider.shape[0])

            if current_k_effective == 0:
                last_y = history[-1][1] if history else np.nan
                history.append((total_evals, last_y))
                updates_skipped += 1
                continue

            best_indices = np.argsort(values_to_consider)[:current_k_effective]
            best_samples_for_update = samples_to_consider[best_indices]
            best_values_for_update = values_to_consider[best_indices]
            current_iter_min_val = best_values_for_update[0]

            improved = False
            if current_iter_min_val < y_min:
                y_min = current_iter_min_val
                x_min = best_samples_for_update[0]
                last_y_min_improvement_iter = iteration
                improved = True

            history.append((total_evals, y_min if np.isfinite(y_min) else np.nan))

            mu = current_k_effective
            weights = np.ones(mu) / mu  
            if weighting == "logrank":
                ranks = np.arange(mu)  
                weights = np.log(mu + 0.5) - np.log(ranks + 1)
                weights = np.maximum(weights, 0)
                weights /= np.sum(weights)
            elif weighting == "softmax":
                vals = best_values_for_update
                vals = np.array(vals)
                weights = np.exp(-(vals - np.min(vals)) / max(temperature, 1e-8))
                weights /= np.sum(weights)
            elif weighting == "exp":
                vals = best_values_for_update
                vals = np.array(vals)
                weights = np.exp(-(vals - np.min(vals)) / max(temperature, 1e-8))
                weights /= np.sum(weights)

            ## MPS update part!!!
            self.MPS.one_site_batch_grad_env_norm_weights_bcktrck(
                        samples=best_samples_for_update,
                        weights=weights,
                        steps_per_site=1,
                        num_sweeps=self.num_sweeps,
                        learning_rate=self.learning_rate
                    )
            
            # self.MPS.one_site_batch_grad_weighted_default(
            #             samples=best_samples_for_update,
            #             num_sweeps=self.num_sweeps,
            #             steps_per_site=1,
            #             learning_rate=self.learning_rate,
            #             weights=weights,
            #         )
        
            if iteration % 5 == 0 or iteration == 1 or improved:
                loss = -np.mean(self.MPS.log_likelihood_1norm_samples(best_samples_for_update))          
                in_stagnation_exploration_mode = True
                exploration_iters_left = 5
                last_y_min_improvement_iter = iteration

                y_min_print = y_min if np.isfinite(y_min) else float('nan')
                print(f"    Iter {iteration} [Evals {total_evals}/{self.num_bbox_func_query}]: Best y_min = {y_min_print:.3e}  LL = {loss:.3e}  mut={mut_rate:.3f}")

        end_time = time.time()
        print(f"--- Finished Optimization Engine ({(end_time - start_time):.2f} sec) ---")
        print(f"    Best value y_min: {y_min}")
        print(f"    Best configuration x_min: {x_min}")
        return x_min, y_min, history

    def opt_engine_weighted(self, base_seed=None, weighting="logrank", temperature=0.5):
       
        if base_seed is None:
            base_seed = int.from_bytes(np.random.bytes(4), byteorder='big')
        rng = np.random.RandomState(base_seed)

        y_min = np.inf
        x_min = None
        history = []
        updates_skipped = 0
        start_time = time.time()

        print(f"\n--- Running Optimization (weighted, scheme={weighting}) ---")
        print(f"    Budget M={self.num_bbox_func_query}, K={self.tot_num_samples}, k={self.num_top_samples}")
        print(f"    Update params: lr={self.learning_rate}, sweeps={self.num_sweeps}, mut={self.base_mutation_rate}")
        if weighting in ("softmax", "exp"):
            print(f"    Weighting temperature: {temperature}")

        total_evals = 0
        iteration = 0

        while total_evals < self.num_bbox_func_query:
            iteration += 1
            iter_seed = rng.randint(0, 2**32 - 1)

            samples_to_take = min(self.tot_num_samples, self.num_bbox_func_query - total_evals)
            if samples_to_take <= self.num_top_samples:
                print(f"    Warning: Remaining budget ({self.num_bbox_func_query - total_evals}) <= k ({self.num_top_samples}). Stopping early.")
                break

            current_samples_np = self.MPS.seq_sample_from(num_samples=samples_to_take, seed=iter_seed)
            current_samples_np = self.mutate_samples(current_samples_np, mutation_rate=self.base_mutation_rate, rng=rng)

            sample_values_np = self.bbox_func_eval_samples(current_samples_np)
            total_evals += samples_to_take

            finite_mask = np.isfinite(sample_values_np)
            if not np.any(finite_mask):
                last_y = history[-1][1] if history else np.nan
                history.append((total_evals, last_y))
                updates_skipped += 1
                continue

            samples_to_consider = current_samples_np[finite_mask]
            values_to_consider = sample_values_np[finite_mask]
            current_k_effective = min(self.num_top_samples, samples_to_consider.shape[0])

            if current_k_effective == 0:
                last_y = history[-1][1] if history else np.nan
                history.append((total_evals, last_y))
                updates_skipped += 1
                continue

            best_indices = np.argsort(values_to_consider)[:current_k_effective]
            best_samples_for_update = samples_to_consider[best_indices]
            best_values_for_update = values_to_consider[best_indices]
            current_iter_min_val = best_values_for_update[0]

            if current_iter_min_val < y_min:
                y_min = current_iter_min_val
                x_min = best_samples_for_update[0]

            history.append((total_evals, y_min if np.isfinite(y_min) else np.nan))

            mu = current_k_effective
            weights = np.ones(mu) / mu 
            if weighting == "logrank":
                ranks = np.arange(mu) 
                weights = np.log(mu + 0.5) - np.log(ranks + 1)
                weights = np.maximum(weights, 0)
                weights /= np.sum(weights)
            elif weighting == "softmax":
                vals = best_values_for_update
                vals = np.array(vals)
                weights = np.exp(-(vals - np.min(vals)) / max(temperature, 1e-8))
                weights /= np.sum(weights)
            elif weighting == "exp":
                vals = best_values_for_update
                vals = np.array(vals)
                weights = np.exp(-(vals - np.min(vals)) / max(temperature, 1e-8))
                weights /= np.sum(weights)

            self.MPS.one_site_batch_grad_env_norm_weights_bcktrck(
                        samples=best_samples_for_update,
                        weights=weights,
                        steps_per_site=1,
                        num_sweeps=self.num_sweeps,
                        learning_rate=self.learning_rate
                    )
            
            # self.MPS.one_site_batch_grad_default(
            #             samples=best_samples_for_update,
            #             num_sweeps=self.num_sweeps,
            #             learning_rate=self.learning_rate
            #         )
            
            if iteration % 5 == 0 or iteration == 1:
                loss = -np.mean(self.MPS.log_likelihood_1norm_samples(best_samples_for_update))
                y_min_print = y_min if np.isfinite(y_min) else float('nan')
                print(f"    Iter {iteration} [Evals {total_evals}/{self.num_bbox_func_query}]: Best y_min = {y_min_print:.3e}  LL = {loss:.3e}")

        end_time = time.time()
        print(f"--- Finished Optimization Engine ({(end_time - start_time):.2f} sec) ---")
        print(f"    Best value y_min: {y_min}")
        print(f"    Best configuration x_min: {x_min}")
        return x_min, y_min, history

    def opt_engine_born(self, base_seed=None):
        
        if base_seed is None:
            base_seed = int.from_bytes(np.random.bytes(4), byteorder='big')
        rng = np.random.RandomState(base_seed)

        y_min = np.inf
        x_min = None
        history = []
        updates_skipped = 0
        start_time = time.time()

        print(f"\n--- Running Optimization (EDA-like with Stagnation Response) ---")
        print(f"    Budget M={self.num_bbox_func_query}, K={self.tot_num_samples}, k_update={self.num_top_samples}") 
        print(f"    Update params: lr={self.learning_rate}, sweeps={self.num_sweeps}")
        print(f"    Stagnation: Patience={self.stagnation_patience}, ActiveMutation={self.stagnation_active_mutation_rate}, PerturbStrength={self.perturbation_strength_on_stagnation}")

        total_evals = 0
        iteration = 0
        self.last_y_min_improvement_iter = 0 
        self.in_stagnation_exploration_mode = False 

        while total_evals < self.num_bbox_func_query:
            iteration += 1
            iter_seed = rng.randint(0, 2**32 - 1)

            if not self.in_stagnation_exploration_mode and \
            (iteration - self.last_y_min_improvement_iter > self.stagnation_patience) and \
            (y_min != np.inf): 
                print(f"    Iter {iteration}: Stagnation detected (no improvement in y_min for {self.stagnation_patience} iterations).")
                print(f"    Activating enhanced exploration: Higher mutation & MPS perturbation.")
                self.in_stagnation_exploration_mode = True
                # self.MPS.gaussian_perturbation(sigma=self.perturbation_strength_on_stagnation)
                print(f"    Applied MPS perturbation with strength {self.perturbation_strength_on_stagnation}.")
                self.last_y_min_improvement_iter = iteration 


            current_mutation_to_apply = self.stagnation_active_mutation_rate if self.in_stagnation_exploration_mode else self.base_mutation_rate
            samples_to_take = min(self.tot_num_samples, self.num_bbox_func_query - total_evals)
            if samples_to_take <= self.num_top_samples:
                if total_evals < self.num_bbox_func_query and samples_to_take > 0:
                    print(f"    Warning: Remaining budget ({self.num_bbox_func_query - total_evals}) <= k_update ({self.num_top_samples_eda}). Evaluating remaining samples but stopping MPS updates.")
                    current_samples_np = self.MPS.born_sample(num_samples=samples_to_take, seed=iter_seed)
                    current_samples_np = self.mutate_samples(current_samples_np, mutation_rate=current_mutation_to_apply) 
                    sample_values_np = self.bbox_func_eval_samples(current_samples_np)
                    total_evals += samples_to_take
                    
                    finite_mask_final = np.isfinite(sample_values_np)
                    if np.any(finite_mask_final):
                        current_iter_min_val_final = np.min(sample_values_np[finite_mask_final])
                        if current_iter_min_val_final < y_min:
                            y_min = current_iter_min_val_final
                            best_idx_final = np.argmin(sample_values_np) 
                            x_min = current_samples_np[best_idx_final]
                    
                    if np.isfinite(y_min): history.append((total_evals, y_min))

                print(f"    Stopping optimization loop due to insufficient budget for further meaningful iterations.")
                break


            current_samples_np = self.MPS.born_sample(num_samples=samples_to_take, seed=iter_seed)
            current_samples_np = self.mutate_samples(current_samples_np, mutation_rate=current_mutation_to_apply) 
            sample_values_np = self.bbox_func_eval_samples(current_samples_np)
            total_evals += samples_to_take
            finite_mask = np.isfinite(sample_values_np)
            if not np.any(finite_mask): 
                last_y_min_val = history[-1][1] if history else np.nan
                history.append((total_evals, last_y_min_val))
                updates_skipped += 1
                print(f"    Iter {iteration}: All samples in batch had non-finite costs. Skipping MPS update.")
                continue 

            samples_to_consider = current_samples_np[finite_mask]
            values_to_consider = sample_values_np[finite_mask]


            ##### COLLAPSE CHECK ######
            # I want to do a check here to see if the MPS has collapsed to a single configuration, because in that case the sampling will produce identical samples and mode collapse takes place
            # Especially this causes NLL to approach identially zero, and the optimization gets stuck, no exploration nothing!!
            ###### COLLAPSE CHECK ENDS ######

            current_k_effective = min(self.num_top_samples, samples_to_consider.shape[0])
            if current_k_effective == 0: # Should not happen if np.any(finite_mask) was true
                last_y_min_val = history[-1][1] if history else np.nan
                history.append((total_evals, last_y_min_val))
                updates_skipped += 1
                print(f"    Iter {iteration}: Effective k is 0 after filtering. Skipping MPS update.")
                continue

            best_indices = np.argsort(values_to_consider)[:current_k_effective]
            best_samples_for_update = samples_to_consider[best_indices]
            current_iter_min_val = values_to_consider[best_indices[0]]

            y_min_improved_this_iteration = False
            if current_iter_min_val < y_min:
                y_min = current_iter_min_val
                x_min = samples_to_consider[best_indices[0]] 
                self.last_y_min_improvement_iter = iteration
                y_min_improved_this_iteration = True
                if self.in_stagnation_exploration_mode:
                    print(f"    Iter {iteration}: Improvement found during exploration! Best y_min = {y_min:.4f}. Returning to base mutation.")
                    self.in_stagnation_exploration_mode = False 

            history.append((total_evals, y_min if np.isfinite(y_min) else np.nan))

            # GRADD
            update_result = self.MPS.two_site_batch_grad_born(
                    samples=best_samples_for_update,
                    num_sweeps=self.num_sweeps,
                    learning_rate=self.learning_rate)
                
                # update_result = self.MPS.one_site_tsgd_born(
                #     samples=best_samples_for_update,
                #     num_sweeps=self.num_sweeps,
                #     learning_rate=self.learning_rate)

                # self.MPS.one_site_proj_fit(samples = best_samples_for_update)
            
            if iteration % 5 == 0 or iteration == 1 or y_min_improved_this_iteration or self.in_stagnation_exploration_mode:
                y_min_print = y_min if np.isfinite(y_min) else float('nan')
                loglikes = self.MPS.log_likelihood_2norm_samples(best_samples_for_update)
                # print(best_samples_for_update)
                print('normal or not',self.MPS.braket())
                loss_val = -np.mean(loglikes) 
                print(f"    Iter {iteration} [Evals {total_evals}/{self.num_bbox_func_query}]: Best y_min = {y_min_print:.3e}  NLL_Loss = {loss_val:.4f} Mut={current_mutation_to_apply:.2f}")

       
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"--- Finished EDA Optimization Engine ({(elapsed_time):.2f} sec) ---")
        print(f"    Completed {iteration} iterations.")
        print(f"    Total function evaluations: {total_evals}")
        print(f"    Updates potentially skipped (no/few finite samples): {updates_skipped}")
        print(f"    Best value found (y_min): {y_min}")
        print(f"    Best configuration (x_min): {x_min.tolist() if x_min is not None else None}") 

        return x_min, y_min, history

    
    