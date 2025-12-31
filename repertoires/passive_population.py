"""
PassivePopulationArchive: A simple archive that stores a fixed-size population
and overwrites solutions at each iteration (like gradient descent / Adam optimization).

Works with dictionary-based solutions (same schema as DNS/CVT archives).
"""

import numpy as np
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._utils import parse_dtype
from typing import Dict, Optional, Union


class PassivePopulationArchive(ArchiveBase):
    """
    A passive population archive that stores a fixed population size.
    
    At each iteration (via add), the new solutions overwrite the old ones,
    simulating a simple population update (no selection, no elitism).
    
    Supports dictionary-based solutions for flexibility with different genotype types.
    
    Args:
        solution_dim: Dict mapping solution keys to shapes, e.g. {'latent': (4,64,64), 'prompt_embedding': (77,768)}
        measure_dim: Dimensionality of the descriptors (measures).
        population_size: Number of solutions to maintain.
        dtype: Data type for arrays.
        seed: Random seed.
        extra_fields: Additional fields to store per solution.
        qd_score_offset: Offset for QD score calculation.
        batch_sizes: List of batch sizes for each emitter, e.g. [5, 5, 6].
    """

    def __init__(
        self,
        *,
        solution_dim,
        measure_dim,
        population_size: int,
        dtype=np.float64,
        seed=None,
        extra_fields=None,
        qd_score_offset: float = 0.0,
        batch_sizes=None,
        # Allow specific dtypes if needed, matching the pattern
        solution_dtype=None,
        objective_dtype=None,
        measures_dtype=None,
    ):
        """Initialize the PassivePopulationArchive."""
        # Pass dummy values to base class
        super().__init__(
            solution_dim=(),
            objective_dim=(),
            measure_dim=measure_dim,
        )

        self._population_size = population_size
        self._rng = np.random.default_rng(seed)
        self._qd_score_offset = qd_score_offset
        self._batch_sizes = batch_sizes or []
        self._extra_fields = extra_fields or {}
        
        # --- 1. DTYPE HANDLING (The Fix) ---
        default_dtype = dtype if dtype is not None else np.float64
        
        def get_dt(target, key):
            if target is not None: return target
            if isinstance(default_dtype, dict): return default_dtype.get(key, np.float64)
            return default_dtype

        self._dtypes = {
            "solution": get_dt(solution_dtype, "solution"),
            "objective": get_dt(objective_dtype, "objective"),
            "measures": get_dt(measures_dtype, "measures"),
        }
        
        self._field_list = ["solution", "objective", "measures"] + list(self._extra_fields.keys())

        # --- 2. INDEX HANDLING ---
        self._emitter_start_idx = 0
        self._emitter_end_idx = population_size
        self.start_idxs = []
        self.end_idxs = []
        if self._batch_sizes and len(self._batch_sizes) > 0:
            cum_sum = 0
            for size in self._batch_sizes:
                self.start_idxs.append(cum_sum)
                cum_sum += size
                self.end_idxs.append(cum_sum)
        else:
            self.start_idxs.append(0)
            self.end_idxs.append(population_size)

        # --- 3. SOLUTION SCHEMA HANDLING ---
        if isinstance(solution_dim, dict):
            self._solution_schema = solution_dim
            self._is_structured = True
        else:
            shape = (solution_dim,) if isinstance(solution_dim, int) else solution_dim
            self._solution_schema = {"data": shape}
            self._is_structured = False

        # --- 4. ALLOCATE STORAGE ---
        self._solutions = {}
        for key, shape in self._solution_schema.items():
            self._solutions[key] = np.empty(
                (self._population_size, *shape),
                dtype=self._dtypes["solution"]
            )

        self._objectives = np.full(
            (self._population_size,), -np.inf, dtype=self._dtypes["objective"]
        )
        self._measures = np.empty(
            (self._population_size, self.measure_dim), dtype=self._dtypes["measures"]
        )

        self._extra_field_arrays = {}
        for name, (shape, dt) in self._extra_fields.items():
            self._extra_field_arrays[name] = np.empty(
                (self._population_size, *shape), dtype=dt
            )

        self._num_elites = 0
        self._stats_reset()

    def _stats_reset(self):
        """Resets the archive stats."""
        self._best_elite = None
        self._stats = ArchiveStats(
            num_elites=0,
            coverage=0.0,
            qd_score=0.0,
            norm_qd_score=0.0,
            obj_max=None,
            obj_mean=None,
        )

    def _stats_update(self):
        """Updates statistics after an add operation."""
        if self.empty:
            self._stats_reset()
            return

        occupied_obj = self._objectives[:self._num_elites]
        objective_sum = np.sum(occupied_obj)

        best_idx = np.argmax(occupied_obj)

        # Construct best elite dict
        best_sol = {k: arr[best_idx] for k, arr in self._solutions.items()}
        if not self._is_structured:
            best_sol = best_sol["data"]

        self._best_elite = {
            "solution": best_sol,
            "objective": self._objectives[best_idx],
            "measures": self._measures[best_idx],
        }

        new_obj_max = self._objectives[best_idx]
        new_qd_score = (objective_sum - self._qd_score_offset * self._num_elites)

        self._stats = ArchiveStats(
            num_elites=self._num_elites,
            coverage=float(self._num_elites) / self._population_size,
            qd_score=new_qd_score,
            norm_qd_score=new_qd_score / self._population_size,
            obj_max=new_obj_max,
            obj_mean=objective_sum / self._num_elites,
        )

    def add(
        self,
        solution: Dict,
        objective: np.ndarray,
        measures: np.ndarray,
        **fields,
    ) -> Dict:
        """
        Overwrite the emitter's slice of the population with new solutions (passive update).

        Args:
            solution: Dict of solution arrays (batch_size, *shape).
            objective: Batch of fitness values (batch_size,).
            measures: Batch of descriptors (batch_size, measure_dim).
            **fields: Additional fields.

        Returns:
            add_info: Status dict for each solution.
        """
        
        # Determine number of objectives
        if objective.ndim == 0:
            objective = np.array([objective])
            
        # Normalize solution input
        if not isinstance(solution, dict):
            if self._is_structured:
                raise ValueError("Archive expects a dictionary for 'solution'")
            solution_batch = {"data": solution}
        else:
            if not self._is_structured:
                raise ValueError("Archive created with flat dim, but got dict solution")
            solution_batch = solution

        add_info = {
            "status": np.zeros(objective.shape[0], dtype=np.int32),
            "value": objective.copy(),
        }

        # Overwrite emitter's slice with new solutions
        for k in self._solutions:
            self._solutions[k] = solution_batch[k]

        self._objectives = objective
        self._measures = measures

        # Handle extra fields
        for name, value in fields.items():
            if name in self._extra_field_arrays:
                self._extra_field_arrays[name] = value

        # Mark all as "added"
        add_info["status"][:] = 2

        self._stats_update()

        return add_info

    def sample_elites(self, emitter_id: int) -> Dict:
        """
        Return n elites from the emitter's slice (deterministic, no random sampling).
        
        If n equals the population_size, returns all elites.
        Otherwise, returns the deterministic slice [start_idx:start_idx+n].
        """
        if self.empty:
            raise IndexError("No elites to sample.")

        # Return deterministic slice for this emitter
        slice_size = min(self._population_size, self.end_idxs[emitter_id] - self.start_idxs[emitter_id], self._num_elites)
        all_indices = np.arange(self.start_idxs[emitter_id], self.start_idxs[emitter_id] + slice_size)

        sampled_sol = {k: self._solutions[k][all_indices] for k in self._solutions}

        if not self._is_structured:
            sampled_sol = sampled_sol["data"]

        return {
            "solution": sampled_sol,
            "objective": self._objectives[all_indices],
            "measures": self._measures[all_indices],
            **{k: arr[all_indices] for k, arr in self._extra_field_arrays.items()},
        }

    # --- PROPERTIES ---

    @property
    def stats(self) -> ArchiveStats:
        """ArchiveStats: Statistics about the archive."""
        return self._stats

    @property
    def best_elite(self) -> Dict:
        """dict: The best elite found so far."""
        return self._best_elite

    @property
    def dtypes(self) -> Dict:
        """dict: Mapping from field name to dtype."""
        return self._dtypes

    @property
    def field_list(self) -> list:
        """list: List of data fields stored in the archive."""
        return self._field_list

    @property
    def empty(self) -> bool:
        """bool: True if the archive is empty."""
        return self._num_elites == 0

    @property
    def solution_dim(self) -> Dict:
        """Returns the solution schema (dict)."""
        return self._solution_schema

    def data(self, fields=None, return_type="dict"):
        """
        Returns data for all solutions in the population.

        Args:
            fields: List of fields to retrieve. If None, all fields returned.
            return_type: Format for return ('dict', 'tuple', 'dict_array', 'pandas').

        Returns:
            Data in the requested format.
        """
        # 1. Handle Empty Archive
        if self.empty:
            empty_solutions = {
                k: np.empty((0, *shape), dtype=self.dtypes["solution"])
                for k, shape in self.solution_dim.items()
            }

            all_data = {
                "solution": empty_solutions,
                "objective": np.empty(0, dtype=self.dtypes["objective"]),
                "measures": np.empty((0, self.measure_dim), dtype=self.dtypes["measures"]),
                **{
                    name: np.empty((0, *shape), dtype=dt)
                    for name, (shape, dt) in self._extra_fields.items()
                },
            }
        else:
            # 2. Handle Populated Archive
            sliced_solutions = {
                k: arr[:self._num_elites].copy() for k, arr in self._solutions.items()
            }

            all_data = {
                "solution": sliced_solutions,
                "objective": self._objectives[:self._num_elites].copy(),
                "measures": self._measures[:self._num_elites].copy(),
                **{
                    name: arr[:self._num_elites].copy()
                    for name, arr in self._extra_field_arrays.items()
                },
            }

        # 3. Filter Fields
        if fields is None:
            fields = self.field_list
        elif isinstance(fields, str):
            return all_data[fields]

        # 4. Return Types
        if return_type == "dict":
            return {f: all_data[f] for f in fields}

        if return_type == "dict_array":
            # Used to get single elite dicts
            results = []
            for i in range(self._num_elites):
                elite = {}
                for field in fields:
                    if field == "solution":
                        elite["solution"] = {
                            k: all_data["solution"][k][i] for k in all_data["solution"]
                        }
                    else:
                        elite[field] = all_data[field][i]
                results.append(elite)
            return results

        if return_type == "tuple":
            return tuple(all_data[f] for f in fields)

        if return_type == "pandas":
            try:
                import pandas as pd
            except ImportError as e:
                raise ImportError("pandas is required for return_type='pandas'") from e

            data_dict = {}
            for f in fields:
                val = all_data[f]
                if f == "solution":
                    for sol_k, sol_v in val.items():
                        data_dict[f"solution_{sol_k}"] = list(sol_v)
                elif f == "objective":
                    data_dict["objective"] = val
                elif isinstance(val, np.ndarray) and val.ndim == 2:
                    for i in range(val.shape[1]):
                        data_dict[f"{f}_{i}"] = val[:, i]
                else:
                    data_dict[f] = list(val)
            return pd.DataFrame(data_dict)
