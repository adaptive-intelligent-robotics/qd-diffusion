"""Implementation of sep-CMA-ES that can be used across various emitters.

Adapted from Nikolaus Hansen's pycma:
https://github.com/CMA-ES/pycma/blob/master/cma/purecma.py
"""

import warnings

import numba as nb
import numpy as np

from ribs._utils import readonly
from ribs.emitters.opt._evolution_strategy_base import (
    BOUNDS_SAMPLING_THRESHOLD,
    BOUNDS_WARNING,
    EvolutionStrategyBase,
)

from ribs.emitters._emitter_base import EmitterBase

class CMAESDictEmitter(EmitterBase):
    """
    Wrapper that allows SeparableCMAESDict to be used with a DictScheduler.
    """
    def __init__(
        self,
        archive,
        *,
        initial_solution,
        mutation_key,
        sigma0,
        batch_size,
        num_parents,
        seed=None,
        bounds=None,
    ):
        # We handle the solution schema manually, so we pass solution_dim=() to base
        self._solution_schema = archive.solution_dim
        self._num_parents = num_parents
        super().__init__(archive=archive, solution_dim=(), bounds=bounds)
        
        # Instantiate the Optimizer (The class you provided in the previous turn)
        self.es = SeparableCMAESDict(
            initial_solution=initial_solution,
            mutation_key=mutation_key,
            sigma0=sigma0,
            batch_size=batch_size,
            seed=seed,
            lower_bounds=bounds[0] if bounds else -np.inf,
            upper_bounds=bounds[1] if bounds else np.inf,
        )
        self.batch_size = self.es.batch_size
        
    def ask(self):
        # Simply delegate to the optimizer
        return self.es.ask()

    def tell(self, solution, objective, measures, add_info, **fields):
        # Ribs schedulers provide objectives in the order of the batch.
        # We need to sort them because the ES expects ranked inputs.
        
        # 1. Sort indices by fitness (descending)
        indices = np.argsort(objective)[::-1]
        
        # 2. Get the sorted values
        sorted_values = objective[indices]
        
        # 3. Tell the optimizer (using the top 50% as parents is standard)
        self.es.tell(
            ranking_indices=indices,
            ranking_values=sorted_values,
            num_parents=self._num_parents
        )


class DiagonalMatrix:
    """Maintains a diagonal covariance matrix.

    Args:
        dimension (int): Size of the (square) covariance matrix.
        dtype (str or data-type): Data type of the matrix, typically np.float32 or
            np.float64.
    """

    def __init__(self, dimension, dtype):
        self.cov = np.ones((dimension,), dtype=dtype)
        self.dtype = dtype

        # The last evaluation on which the eigensystem was updated.
        self.updated_eval = 0

    @property
    def condition_number(self):
        """Condition number of the covariance matrix."""
        return np.max(self.cov) / np.min(self.cov)

    @property
    def eigenvalues(self):
        """Eigenvalues (equal to covariance matrix since it is diagonal)."""
        return self.cov

    @property
    def invsqrt(self):
        """C^-1/2."""
        return 1 / np.sqrt(self.cov)


class SeparableCMAESDict(EvolutionStrategyBase):
    """
    A dictionary-aware wrapper around Sep-CMA-ES.
    
    It optimizes ONE key (`mutation_key`) in the dictionary as a flat vector, 
    but `ask()` returns a full dictionary batch (reshaped and broadcasted) 
    compatible with your model/archive.
    """

    def __init__(
        self,
        initial_solution,
        mutation_key,
        sigma0,
        batch_size=None,
        seed=None,
        dtype=np.float64,
        lower_bounds=-np.inf,
        upper_bounds=np.inf,
    ):
        # 1. Setup Dictionary Handling
        if mutation_key not in initial_solution:
            raise ValueError(f"mutation_key '{mutation_key}' not in initial_solution")
        
        self.mutation_key = mutation_key
        
        # Parse initial solution to numpy
        self._seed_sol = {}
        for k, v in initial_solution.items():
            # Handle torch tensors if passed
            val = v.detach().cpu().numpy() if hasattr(v, "detach") else np.array(v)
            if val.ndim > 0 and val.shape[0] == 1:
                val = val.squeeze(0)
            self._seed_sol[k] = val
            
        # Calculate dimensions for the optimizer
        target_arr = self._seed_sol[self.mutation_key]
        self._feature_shape = target_arr.shape
        flat_dim = np.prod(self._feature_shape)
        
        # 2. Standard Init (Passing calculated flat_dim)
        self.batch_size = (
            4 + int(3 * np.log(flat_dim)) if batch_size is None else batch_size
        )
        self.sigma0 = sigma0
        self.solution_dim = flat_dim 
        self.dtype = dtype

        self.lower_bounds = np.asarray(lower_bounds, dtype=self.dtype)
        self.upper_bounds = np.asarray(upper_bounds, dtype=self.dtype)

        self._rng = np.random.default_rng(seed)
        
        # self._solutions stores the FLAT vectors for the math to work
        self._solutions = None 

        # Strategy-specific params
        self.current_eval = None
        self.mean = None
        self.sigma = None
        self.pc = None
        self.ps = None
        self.cov = None
        
        # Initialize mean with the flattened seed
        self.reset(self._seed_sol)

    def reset(self, x0):
        self.current_eval = 0
        self.sigma = self.sigma0
        
        # Handle dict input for reset
        if isinstance(x0, dict):
             # Update seed cache if needed
             # (Optional: you might want to update self._seed_sol here)
             flat_x0 = x0[self.mutation_key].flatten()
        else:
             flat_x0 = x0
             
        self.mean = np.array(flat_x0, self.dtype)

        # Setup evolution path variables.
        self.pc = np.zeros(self.solution_dim, dtype=self.dtype)
        self.ps = np.zeros(self.solution_dim, dtype=self.dtype)

        # Setup the covariance matrix.
        self.cov = DiagonalMatrix(self.solution_dim, self.dtype)



    def check_stop(self, ranking_values):
        # Tolerances from pycma CMA-ES.
        if self.cov.condition_number > 1e14:
            return True

        # Area of distribution too small.
        area = self.sigma * np.sqrt(max(self.cov.eigenvalues))
        if area < 1e-11:
            return True

        # Fitness is too flat (only applies if there are at least 2 parents).
        # NOTE: We use norm here because we may have multiple ranking values.
        if (
            len(ranking_values) >= 2
            and np.linalg.norm(ranking_values[0] - ranking_values[-1]) < 1e-12
        ):
            return True

        return False



    @staticmethod
    @nb.jit(nopython=True)
    def _transform_and_check_sol(
        unscaled_params, transform_vec, mean, lower_bounds, upper_bounds
    ):
        """Numba helper for transforming parameters to the solution space."""
        solutions = np.expand_dims(
            transform_vec, axis=0
        ) * unscaled_params + np.expand_dims(mean, axis=0)
        out_of_bounds = np.logical_or(
            solutions < np.expand_dims(lower_bounds, axis=0),
            solutions > np.expand_dims(upper_bounds, axis=0),
        )
        return solutions, out_of_bounds

    def ask(self, batch_size=None):
        """
        Returns a DICTIONARY of solutions (batch, ...).
        """
        if batch_size is None:
            batch_size = self.batch_size

        # 1. Generate FLAT solutions (Standard Logic)
        self._solutions = np.empty((batch_size, self.solution_dim), dtype=self.dtype)
        transform_vec = np.sqrt(self.cov.eigenvalues)

        # Resampling method for bound constraints -> sample new solutions until
        # all solutions are within bounds.
        remaining_indices = np.arange(batch_size)
        sampling_itrs = 0
        while len(remaining_indices) > 0:
            unscaled_params = self._rng.normal(
                0.0,
                self.sigma,
                (len(remaining_indices), self.solution_dim),
            ).astype(self.dtype)
            new_solutions, out_of_bounds = self._transform_and_check_sol(
                unscaled_params,
                transform_vec,
                self.mean,
                self.lower_bounds,
                self.upper_bounds,
            )
            self._solutions[remaining_indices] = new_solutions

            # Find indices in remaining_indices that are still out of bounds
            # (out_of_bounds indicates whether each value in each solution is
            # out of bounds).
            remaining_indices = remaining_indices[np.any(out_of_bounds, axis=1)]

            # Warn if we have resampled too many times.
            sampling_itrs += 1
            if sampling_itrs > BOUNDS_SAMPLING_THRESHOLD:
                warnings.warn(BOUNDS_WARNING)

        # 2. Reshape & Broadcast to Dictionary (New Logic)
        
        # Reshape the mutation key: (Batch, Flat) -> (Batch, *FeatureShape)
        reshaped_mutations = self._solutions.reshape(batch_size, *self._feature_shape)
        
        batch_dict = {}
        for k, v in self._seed_sol.items():
            if k == self.mutation_key:
                batch_dict[k] = reshaped_mutations.astype(np.float32)
            else:
                # Broadcast fixed keys: (Shape) -> (Batch, Shape)
                arr = v[np.newaxis, ...]
                reps = [batch_size] + [1] * (arr.ndim - 1)
                batch_dict[k] = np.repeat(arr, reps[0], axis=0).astype(np.float32)
                
        # We return the dictionary, but self._solutions keeps the flat arrays
        # so that tell() works without modification.
        return batch_dict


    @staticmethod
    def _conedf(df, mu, solution_dim):
        """Used for computing separable learning rate."""
        return 1.0 / (df + 2.0 * np.sqrt(df) + float(mu) / solution_dim)

    @staticmethod
    def _cmudf(df, mu, alphamu):
        """Used for computing separable learning rate."""
        return (alphamu + mu + 1.0 / mu - 2) / (df + 4 * np.sqrt(df) + mu / 2.0)

    def _calc_strat_params(self, solution_dim, num_parents):
        """Calculates weights, mueff, and learning rates for sep-CMA-ES.

        Refer here
        https://github.com/CMA-ES/pycma/blob/master/cma/evolution_strategy.py#L3767
        for the learning rates.
        """
        # Create fresh weights for the number of parents found.
        weights = np.log(num_parents + 0.5) - np.log(np.arange(1, num_parents + 1))
        total_weights = np.sum(weights)
        weights = weights / total_weights
        mueff = np.sum(weights) ** 2 / np.sum(weights**2)

        # Dynamically update these strategy-specific parameters.
        cc_sep = (1 + 1 / solution_dim + mueff / solution_dim) / (
            solution_dim**0.5 + 1 / solution_dim + 2 * mueff / solution_dim
        )

        cs = (mueff + 2) / (solution_dim + mueff + 5)

        c1 = 2 / ((solution_dim + 1.3) ** 2 + mueff)
        c1_sep = c1 * self._conedf(solution_dim, mueff, solution_dim)

        # Instead of 0, pycma uses "rankmu_offset" here but also mentions it
        # barely affects performance. The rest of this code also doesn't use
        # rankmu_offset.
        cmu_sep = min(1 - c1_sep, self._cmudf(solution_dim, mueff, 0))

        return weights, mueff, cc_sep, cs, c1_sep, cmu_sep

    @staticmethod
    @nb.jit(nopython=True)
    def _calc_mean(parents, weights):
        """Numba helper for calculating the new mean."""
        return np.sum(parents * np.expand_dims(weights, axis=1), axis=0)

    @staticmethod
    @nb.jit(nopython=True)
    def _calc_weighted_ys(parents, old_mean, weights):
        """Calculates y's for use in rank-mu update."""
        ys = parents - np.expand_dims(old_mean, axis=0)
        return ys * np.expand_dims(weights, axis=1), ys

    @staticmethod
    @nb.jit(nopython=True)
    def _calc_cov_update(cov, c1a, cmu, c1, pc, sigma, rank_mu_update, weights):
        """Calculates covariance matrix update."""
        rank_one_update = c1 * pc**2
        return (
            cov * (1 - c1a - cmu * np.sum(weights))
            + rank_one_update * c1
            + rank_mu_update * cmu / (sigma**2)
        )

    def tell(self, ranking_indices, ranking_values, num_parents):
        self.current_eval += len(self._solutions[ranking_indices])

        if num_parents == 0:
            return

        parents = self._solutions[ranking_indices][:num_parents]

        weights, mueff, cc, cs, c1, cmu = self._calc_strat_params(
            self.solution_dim, num_parents
        )

        damps = (
            1
            + 2
            * max(
                0,
                np.sqrt((mueff - 1) / (self.solution_dim + 1)) - 1,
            )
            + cs
        )

        # Recombination of the new mean.
        old_mean = self.mean
        self.mean = self._calc_mean(parents, weights)

        # Update the evolution path.
        y = self.mean - old_mean
        z = self.cov.invsqrt * y
        self.ps = (1 - cs) * self.ps + (np.sqrt(cs * (2 - cs) * mueff) / self.sigma) * z
        left = (
            np.sum(np.square(self.ps))
            / self.solution_dim
            / (1 - (1 - cs) ** (2 * self.current_eval / self.batch_size))
        )
        right = 2 + 4.0 / (self.solution_dim + 1)
        hsig = 1 if left < right else 0

        self.pc = (1 - cc) * self.pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y

        # Adapt the covariance matrix.
        weighted_ys, ys = self._calc_weighted_ys(parents, old_mean, weights)
        # Equivalent to calculating the outer product of each ys[i] with itself
        # and taking a weighted sum of the outer products.
        rank_mu_update = np.sum(weighted_ys * ys, axis=0)
        c1a = c1 * (1 - (1 - hsig**2) * cc * (2 - cc))
        self.cov.cov = self._calc_cov_update(
            self.cov.cov, c1a, cmu, c1, self.pc, self.sigma, rank_mu_update, weights
        )

        # Update sigma.
        cn, sum_square_ps = cs / damps, np.sum(np.square(self.ps))
        self.sigma *= np.exp(min(1, cn * (sum_square_ps / self.solution_dim - 1) / 2))