import numpy as np
from numpy.typing import DTypeLike

from ribs._utils import (
    validate_batch,
)
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._utils import (
    parse_dtype,
) 
from scipy.spatial.distance import cdist


class DNSDictArchive(ArchiveBase):
    """
    A DNS Archive that supports dictionary-based solutions (e.g. separate 
    tensors for latents, embeddings, and pooled_embeddings).
    """

    def __init__(
        self,
        *,
        solution_dim, 
        measure_dim,
        max_size,
        fitness_threshold=-np.inf,
        k=3,
        distance_metric="euclidean",
        qd_score_offset=0.0,
        seed=None,
        solution_dtype: DTypeLike = None,
        objective_dtype: DTypeLike = None,
        measures_dtype: DTypeLike = None,
        dtype: DTypeLike = None,
        extra_fields=None,
    ):
        super().__init__(
            solution_dim=(), 
            objective_dim=(),
            measure_dim=measure_dim,
        )

        self._max_size = max_size
        self._fitness_threshold = fitness_threshold
        self._k = k
        self._distance_metric = distance_metric
        self._rng = np.random.default_rng(seed)
        self._qd_score_offset = qd_score_offset
        self._extra_fields = extra_fields or {}

        # --- 1. DTYPE HANDLING ---
        default_dtype = dtype if dtype is not None else np.float64
        
        # Helper to check specific arg -> dict key -> default scalar
        def get_dt(target, key):
            if target is not None: return target
            if isinstance(default_dtype, dict): return default_dtype.get(key, np.float64)
            return default_dtype

        self._dtypes = {
            "solution": get_dt(solution_dtype, "solution"),
            "objective": get_dt(objective_dtype, "objective"),
            "measures": get_dt(measures_dtype, "measures"),
        }

        # --- 2. SCHEMA HANDLING ---
        if isinstance(solution_dim, dict):
            self._solution_schema = solution_dim
            self._is_structured = True
        else:
            shape = (solution_dim,) if isinstance(solution_dim, int) else solution_dim
            self._solution_schema = {"data": shape}
            self._is_structured = False

        self._field_list = ["solution", "objective", "measures"] + list(self._extra_fields.keys())

        # --- 3. STORAGE ALLOCATION (Buffer Style - No ArrayStore) ---
        
        # A. Solution Storage (Parallel Dictionary)
        self._solutions = {}
        for key, shape in self._solution_schema.items():
            self._solutions[key] = np.empty(
                (self._max_size, *shape),
                dtype=self._dtypes["solution"]
            )

        # B. Metadata Storage
        self._objectives = np.full(
            (self._max_size,), -np.inf, dtype=self._dtypes["objective"]
        )
        self._measures = np.empty(
            (self._max_size, self.measure_dim), dtype=self._dtypes["measures"]
        )

        # C. Extra Fields Storage
        self._extra_field_arrays = {}
        for name, (shape, dt) in self._extra_fields.items():
            self._extra_field_arrays[name] = np.empty(
                (self._max_size, *shape), dtype=dt
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
            coverage=float(self._num_elites) / self._max_size,
            qd_score=new_qd_score,
            norm_qd_score=new_qd_score / self._max_size,
            obj_max=new_obj_max,
            obj_mean=objective_sum / self._num_elites,
        )
    
    def add(self, solution, objective, measures, **fields):
        """
        Args:
            solution: A DICT of batches if using structured schema, 
                      or a single numpy array if using standard mode.
        """
        
        # 1. Normalize Solution Input
        if not isinstance(solution, dict):
            if self._is_structured:
                raise ValueError("Archive expects a dictionary for 'solution'")
            solution_batch = {"data": solution}
        else:
            if not self._is_structured:
                raise ValueError("Archive created with flat dim, but got dict solution")
            solution_batch = solution

        if objective.ndim == 0:
            objective = np.array([objective])
            
        batch_size = len(objective)
        
        # Custom validation (skipping ribs validate_batch for solution part)
        if len(measures) != batch_size:
            raise ValueError("Shape mismatch between objective and measures")

        # Add info tracking
        add_info = {
            "status": np.zeros(batch_size, dtype=np.int32),
            "value": np.full(batch_size, np.nan, dtype=self._dtypes["objective"]),
        }
        
        # 2. Thresholding
        candidate_mask = np.ones(batch_size, dtype=bool)
        if not self.empty:
            candidate_mask = objective > self._fitness_threshold
        
        if np.sum(candidate_mask) == 0:
            return add_info

        # Filter inputs
        cand_obj = objective[candidate_mask]
        cand_meas = measures[candidate_mask]
        cand_sols = {k: v[candidate_mask] for k, v in solution_batch.items()}
        cand_extras = {k: v[candidate_mask] for k, v in fields.items()}

        # 3. Concatenate
        cur_elites = self._num_elites
        
        # Concatenate Solutions (Iterating over the dict keys)
        all_sols = {}
        for k in self._solutions:
            all_sols[k] = np.concatenate(
                (self._solutions[k][:cur_elites], cand_sols[k])
            )

        all_objs = np.concatenate((self._objectives[:cur_elites], cand_obj))
        all_measures = np.concatenate((self._measures[:cur_elites], cand_meas))
        
        all_extras = {}
        for k in self._extra_fields:
            all_extras[k] = np.concatenate(
                (self._extra_field_arrays[k][:cur_elites], cand_extras[k])
            )

        num_all = len(all_objs)

        # 4. DNS Calculation
        fitter = all_objs[:, None] < all_objs[None, :]
        np.fill_diagonal(fitter, False)

        if self._distance_metric == "euclidean":
            distance = cdist(all_measures, all_measures, metric="euclidean")
        elif self._distance_metric == "cosine":
            distance = cdist(all_measures, all_measures, metric="cosine")
            distance = np.clip(distance, 0.0, 2.0)

        distance = np.where(fitter, distance, np.inf)
        k_to_find = min(self._k, num_all - 1)
        
        if k_to_find <= 0:
            dns_score = np.full(num_all, np.inf)
        else:
            k_nearest_indices = np.argpartition(distance, k_to_find, axis=1)[:, :k_to_find]
            k_nearest_dists = np.take_along_axis(distance, k_nearest_indices, axis=1)
            k_nearest_dists[np.isinf(k_nearest_dists)] = np.nan
            dns_score = np.nanmean(k_nearest_dists, axis=1)
            dns_score[np.isnan(dns_score)] = np.inf
        
        # 5. Sort and Truncate
        sorted_indices = np.argsort(dns_score)[::-1]
        keep_indices = sorted_indices[:self._max_size]
        new_num_elites = len(keep_indices)

        # 6. Update Status info
        candidate_indices_in_all = np.arange(cur_elites, num_all)
        survived = np.isin(candidate_indices_in_all, keep_indices)
        
        add_info["status"][candidate_mask] = np.where(survived, 2, 0)
        add_info["value"][candidate_mask] = np.where(survived, cand_obj, np.nan)

        # 7. Update Archive Storage
        # Update every tensor in the solution dict
        for k in self._solutions:
            self._solutions[k][:new_num_elites] = all_sols[k][keep_indices]

        self._objectives[:new_num_elites] = all_objs[keep_indices]
        self._measures[:new_num_elites] = all_measures[keep_indices]
        
        for k in self._extra_fields:
            self._extra_field_arrays[k][:new_num_elites] = all_extras[k][keep_indices]

        self._num_elites = new_num_elites
        self._objectives[self._num_elites:] = -np.inf
        
        self._stats_update()
        
        return add_info

    def sample_elites(self, n):
        if self.empty:
            raise IndexError("No elites to sample.")
            
        indices = self._rng.integers(self._num_elites, size=n)
        
        # Retrieve structured solution
        sampled_sol = {
            k: self._solutions[k][indices] for k in self._solutions
        }
        
        # If user initialized with simple tuple/int, return just the array
        if not self._is_structured:
            sampled_sol = sampled_sol["data"]

        return {
            "solution": sampled_sol,
            "objective": self._objectives[indices],
            "measures": self._measures[indices],
            **{k: arr[indices] for k, arr in self._extra_field_arrays.items()}
        }
        
    ##
    ## Properties
    ##

    @property
    def stats(self):
        """ArchiveStats: Statistics about the archive (coverage, qd_score, etc)."""
        return self._stats

    @property
    def best_elite(self):
        """dict: The best elite found so far."""
        return self._best_elite

    @property
    def dtypes(self):
        """dict: Mapping from field name to dtype."""
        return self._dtypes

    @property
    def field_list(self):
        """list: List of data fields stored in the archive."""
        return self._field_list
    
    @property
    def empty(self):
        return self._num_elites == 0
    
    @property
    def solution_dim(self):
        """Returns the solution schema (dict) instead of the raw tuple."""
        return self._solution_schema

    def data(self, fields=None, return_type="dict"):
        """Returns data for all elites in the archive."""
        
        # 1. Handle Empty Archive
        if self.empty:
            # Create a Dict of empty arrays for the solution
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
                }
            }
        else:
            # 2. Handle Populated Archive
            # Iterate over keys and slice the individual arrays
            sliced_solutions = {
                k: arr[:self._num_elites].copy()
                for k, arr in self._solutions.items()
            }

            all_data = {
                "solution": sliced_solutions,
                "objective": self._objectives[:self._num_elites].copy(),
                "measures": self._measures[:self._num_elites].copy(),
                **{
                    name: arr[:self._num_elites].copy()
                    for name, arr in self._extra_field_arrays.items()
                }
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
            # Used by _stats_update to get single elite dicts
            results = []
            for i in range(self._num_elites):
                elite = {}
                for field in fields:
                    if field == "solution":
                        # Extract the i-th item from every key in the solution dict
                        elite["solution"] = {
                            k: all_data["solution"][k][i] 
                            for k in all_data["solution"]
                        }
                    else:
                        elite[field] = all_data[field][i]
                results.append(elite)
            return results

        if return_type == "tuple":
            return tuple(all_data[f] for f in fields)
        
        if return_type == "pandas":
            # Flatten nested solution dicts into columns
            try:
                import pandas as pd
            except ImportError as e:
                raise ImportError("pandas is required for return_type='pandas'") from e
            
            data_dict = {}
            for f in fields:
                val = all_data[f]
                if f == "solution":
                    # Flatten solution dict: solution_latent, solution_prompt...
                    for sol_k, sol_v in val.items():
                        # High-dim tensors are stored as object columns
                        data_dict[f"solution_{sol_k}"] = list(sol_v)
                elif f == "objective":
                    data_dict["objective"] = val
                elif isinstance(val, np.ndarray) and val.ndim == 2:
                    for i in range(val.shape[1]):
                        data_dict[f"{f}_{i}"] = val[:, i]
                else:
                    data_dict[f] = list(val)
            return pd.DataFrame(data_dict)
            
        raise ValueError(f"Unknown return_type: {return_type}")
    
class DNSArchive(ArchiveBase):
    """An unstructured archive that stores a fixed-size buffer of elites.

    This archive implements the (D)iversity-driven (N)ovelty (S)earch archive,
    described in https://dl.acm.org/doi/pdf/10.1145/3712256.3726310.

    It is an unstructured archive, meaning it does not discretize the
    measure space into cells. Instead, it maintains a buffer of up to
    `max_size` elites.

    When new solutions are added, they are concatenated with the existing
    elites. Then, a score is computed for every solution (old and new)
    based on the mean distance to its `k` "fitter" nearest neighbors
    in the measure space. The solutions are sorted by this score,
    and only the top `max_size` solutions are retained.

    Since this is an unstructured archive, the `retrieve()` and
    `retrieve_single()` methods are not implemented. Use `sample_elites()`
    to retrieve solutions.

    Args:
        solution_dim (int or tuple of int): Dimensionality of the solution space.
        measure_dim (int): Dimensionality of the measure space.
        max_size (int): The maximum number of elites to store in the archive.
        k (int): The number of "fitter" nearest neighbors to consider when
            calculating the DNS score. Defaults to 3 (as in the QDAX impl).
        distance_metric (str): The metric to use for calculating distance
            in the measure space. Can be "euclidean" (default) or "cosine".
        qd_score_offset (float): An offset subtracted from all objectives
            when calculating the QD score.
        seed (int): Value to seed the random number generator.
        dtype (str or data-type or dict): Data type of the solutions,
            objectives, and measures. See `parse_dtype` for details.
        extra_fields (dict): Description of extra fields to store. Maps
            field name (str) to a tuple of `(shape, dtype)`.
    """

    def __init__(
        self,
        *,
        solution_dim,
        measure_dim,
        max_size,
        fitness_threshold=-np.inf,
        k=3,
        distance_metric="euclidean",
        qd_score_offset=0.0,
        seed=None,
        dtype=np.float64,
        extra_fields=None,
    ):
        # Call the base constructor. DNS is single-objective, so objective_dim=().
        super().__init__(
            solution_dim=solution_dim,
            objective_dim=(),
            measure_dim=measure_dim,
        )

        # Archive parameters
        self._max_size = max_size
        self._fitness_threshold = fitness_threshold
        self._k = k
        if distance_metric not in ["euclidean", "cosine"]:
            raise ValueError(
                f"Unknown distance_metric: {distance_metric}. "
                "Must be 'euclidean' or 'cosine'."
            )
        self._distance_metric = distance_metric
        
        self._rng = np.random.default_rng(seed)
        self._qd_score_offset = qd_score_offset

        # Handle dtypes and extra fields
        self._dtypes = parse_dtype(dtype)
        self._extra_fields = extra_fields or {}

        # Define the fields stored in this archive
        self._field_list = [
            "solution", "objective", "measures", *self._extra_fields.keys()
        ]
        
        # Internal solution shape (e.g., () for scalar, (dim,) for 1D)
        if isinstance(self.solution_dim, int):
            self._solution_shape = (self.solution_dim,)
        else:
            self._solution_shape = self.solution_dim
            
        # Initialize the data arrays.
        # We use pre-allocated NumPy arrays instead of ArrayStore because
        # this archive is unstructured and requires sorting.
        self._solutions = np.empty(
            (self._max_size, *self._solution_shape),
            dtype=self._dtypes["solution"]
        )
        self._objectives = np.full(
            (self._max_size,), -np.inf, dtype=self._dtypes["objective"]
        )
        self._measures = np.empty(
            (self._max_size, self.measure_dim), dtype=self._dtypes["measures"]
        )

        # Store extra field data in a dict of arrays
        self._extra_field_arrays = {}
        for name, (shape, dt) in self._extra_fields.items():
            self._extra_field_arrays[name] = np.empty(
                (self._max_size, *shape), dtype=dt
            )
            
        self._num_elites = 0

        # Statistics
        self._best_elite = None
        self._stats = None
        self._stats_reset()

    def _stats_reset(self):
        """Resets the archive stats."""
        self._best_elite = None
        self._stats = ArchiveStats(
            num_elites=0,
            # Coverage is % of buffer filled
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

        # Get data from occupied slots
        occupied_obj = self._objectives[:self._num_elites]
        
        objective_sum = np.sum(occupied_obj)
        
        # Find new best elite
        best_idx = np.argmax(occupied_obj)
        self._best_elite = self.data(return_type="dict_array")[best_idx]
        new_obj_max = self._best_elite["objective"]

        # Calculate QD score with offset
        new_qd_score = (
            objective_sum - 
            self._qd_score_offset * self._num_elites
        )
        
        self._stats = ArchiveStats(
            num_elites=self._num_elites,
            coverage=float(self._num_elites) / self._max_size,
            qd_score=new_qd_score,
            norm_qd_score=new_qd_score / self._max_size,
            obj_max=new_obj_max,
            obj_mean=objective_sum / self._num_elites,
        )

    ##
    ## Properties
    ##

    @property
    def field_list(self):
        return self._field_list

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def stats(self):
        return self._stats

    @property
    def empty(self):
        return self._num_elites == 0
    
    @property
    def best_elite(self):
        return self._best_elite

    @property
    def max_size(self):
        """int: The maximum number of elites in the archive."""
        return self._max_size

    ##
    ## Dunder methods
    ##

    def __len__(self):
        return self._num_elites

    def __iter__(self):
        """Creates an iterator over the elites in the archive."""
        for i in range(self._num_elites):
            yield {
                "solution": self._solutions[i],
                "objective": self._objectives[i],
                "measures": self._measures[i],
                **{
                    name: arr[i]
                    for name, arr in self._extra_field_arrays.items()
                }
            }

    def add(self, solution, objective, measures, **fields):
        """Adds a batch of solutions to the archive.

        If the archive is not empty, solutions are first filtered by
        the `fitness_threshold`. If the archive IS empty, all
        solutions are added to seed the archive.
        
        Remaining solutions are then added, and the DNS culling is applied.
        """
        
        # 1. Validation
        data = validate_batch(
            self,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
        )
        
        batch_size = len(data["objective"])
        
        # Create the add_info dict for the *original* batch
        add_info = {
            "status": np.zeros(batch_size, dtype=np.int32),
            "value": np.full(batch_size, np.nan, dtype=self.dtypes["objective"]),
        }
                
        candidate_mask_in_original_batch = np.ones(batch_size, dtype=bool)

        if not self.empty:
            candidate_mask_in_original_batch = data["objective"] > self._fitness_threshold
        
        num_candidates = np.sum(candidate_mask_in_original_batch)
        
        if num_candidates == 0:
            return add_info

        # Filter all data to only include candidates
        candidate_data = {
            name: arr[candidate_mask_in_original_batch] for name, arr in data.items()
        }
        

        
        # 2. Concatenate (now using `candidate_data`)
        cur_elites = self._num_elites
        
        all_sols = np.concatenate(
            (self._solutions[:cur_elites], candidate_data["solution"])
        )
        all_objs = np.concatenate(
            (self._objectives[:cur_elites], candidate_data["objective"])
        )
        all_measures = np.concatenate(
            (self._measures[:cur_elites], candidate_data["measures"])
        )
        
        all_extra_fields = {}
        for name in self._extra_fields:
            all_extra_fields[name] = np.concatenate(
                (self._extra_field_arrays[name][:cur_elites], candidate_data[name])
            )
        
        num_all = len(all_objs)

        # 3. Calculate "fitter" matrix
        fitter = all_objs[:, None] < all_objs[None, :]
        np.fill_diagonal(fitter, False)

        # 4. Calculate distances (using cdist)
        if self._distance_metric == "euclidean":
            distance = cdist(all_measures, all_measures, metric="euclidean")
        elif self._distance_metric == "cosine":
            distance = cdist(all_measures, all_measures, metric="cosine")
            distance = np.clip(distance, 0.0, 2.0)

        # 5. Compute DNS score
        distance = np.where(fitter, distance, np.inf)
        k_to_find = min(self._k, num_all - 1)
        if k_to_find <= 0:
            dns_score = np.full(num_all, np.inf)
        else:
            k_nearest_indices = np.argpartition(
                distance, k_to_find, axis=1
            )[:, :k_to_find]
            k_nearest_dists = np.take_along_axis(
                distance, k_nearest_indices, axis=1
            )
            k_nearest_dists[np.isinf(k_nearest_dists)] = np.nan
            dns_score = np.nanmean(k_nearest_dists, axis=1)
            dns_score[np.isnan(dns_score)] = np.inf
        
        # 6. Sort and Truncate
        sorted_indices = np.argsort(dns_score)[::-1]
        keep_indices = sorted_indices[:self._max_size]
        new_num_elites = len(keep_indices)

        # 7. Calculate add_info (status)
        
        # Indices of the *candidate* solutions within the `all_` arrays
        candidate_sol_indices_in_all = np.arange(cur_elites, num_all)
        
        # `survived` is a boolean mask of shape (num_candidates,)
        survived = np.isin(candidate_sol_indices_in_all, keep_indices)
        
        # --- MAP SURVIVAL BACK TO ORIGINAL add_info ---
        
        # Create status/value arrays just for the candidates
        candidate_status = np.where(survived, 2, 0).astype(np.int32)
        candidate_value = np.where(survived, candidate_data["objective"], np.nan)

        # "Scatter" these results back into the original add_info
        add_info["status"][candidate_mask_in_original_batch] = candidate_status
        add_info["value"][candidate_mask_in_original_batch] = candidate_value

        # 8. Update Archive State
        self._solutions[:new_num_elites] = all_sols[keep_indices]
        self._objectives[:new_num_elites] = all_objs[keep_indices]
        self._measures[:new_num_elites] = all_measures[keep_indices]
        
        for name in self._extra_fields:
            self._extra_field_arrays[name][:new_num_elites] = \
                all_extra_fields[name][keep_indices]
        
        self._num_elites = new_num_elites
        
        self._objectives[self._num_elites:] = -np.inf
        
        # 9. Update Stats
        self._stats_update()
        
        return add_info

    def add_single(self, solution, objective, measures, **fields):
        """Adds a single solution to the archive.
        
        This is a convenience wrapper around `add()`.
        """
        # Batch all inputs to shape (1, ...)
        solution_batch = np.expand_dims(solution, axis=0)
        objective_batch = np.expand_dims(objective, axis=0)
        measures_batch = np.expand_dims(measures, axis=0)
        
        fields_batch = {
            name: np.expand_dims(arr, axis=0)
            for name, arr in fields.items()
        }
        
        # Call batch add
        add_info = self.add(
            solution_batch,
            objective_batch,
            measures_batch,
            **fields_batch
        )
        
        # Unpack the result from shape (1,) to scalar
        return {
            "status": add_info["status"][0],
            "value": add_info["value"][0],
        }

    def clear(self):
        """Removes all elites from the archive."""
        # Set all objectives to -inf
        self._objectives[:] = -np.inf
        self._num_elites = 0
        self._stats_reset()

    ##
    ## Methods for reading from the archive
    ##

    def retrieve(self, measures):
        """Not implemented for DNSArchive.

        DNSArchive is an unstructured archive and does not map
        measures to specific cells. Therefore, retrieving an
        elite by its measure coordinates is not a supported
        operation.
        
        Use `sample_elites()` to retrieve elites from the archive.
        """
        raise NotImplementedError(
            "`retrieve` is not implemented for DNSArchive. "
            "Use `sample_elites()` instead."
        )

    def retrieve_single(self, measures):
        """Not implemented for DNSArchive.
        
        See `retrieve()` for details.
        """
        raise NotImplementedError(
            "`retrieve_single` is not implemented for DNSArchive. "
            "Use `sample_elites()` instead."
        )

    def data(self, fields=None, return_type="dict"):
        """Returns data for all elites in the archive.

        Args:
            fields (str or array-like of str): List of fields to include.
                Defaults to all fields.
            return_type (str): Type of data to return.
                - "dict": (default) Dict mapping field name to array.
                - "dict_array": Dict mapping field name to array, but
                  for internal use (returns dicts of single elites).
                - "tuple": Tuple of arrays in field order.
                - "pandas": A pandas DataFrame (requires pandas).

        Returns:
            The data for all elites (see `return_type`).
        """
        if self.empty:
            # Return empty data structures
            all_data = {
                "solution": np.empty((0, *self._solution_shape), dtype=self.dtypes["solution"]),
                "objective": np.empty(0, dtype=self.dtypes["objective"]),
                "measures": np.empty((0, self.measure_dim), dtype=self.dtypes["measures"]),
                **{
                    name: np.empty((0, *shape), dtype=dt)
                    for name, (shape, dt) in self._extra_fields.items()
                }
            }
        else:
            # Return copies of the occupied data
            all_data = {
                "solution": self._solutions[:self._num_elites].copy(),
                "objective": self._objectives[:self._num_elites].copy(),
                "measures": self._measures[:self._num_elites].copy(),
                **{
                    name: arr[:self._num_elites].copy()
                    for name, arr in self._extra_field_arrays.items()
                }
            }

        if fields is None:
            fields = self.field_list
        elif isinstance(fields, str):
            return all_data[fields]
            
        # Handle different return types
        if return_type == "dict":
            return {f: all_data[f] for f in fields}
        
        if return_type == "dict_array":
            # Used by _stats_update to get single elite dicts
            return [
                {field: all_data[field][i] for field in fields}
                for i in range(self._num_elites)
            ]

        if return_type == "tuple":
            return tuple(all_data[f] for f in fields)
        
        if return_type == "pandas":
            try:
                import pandas as pd
            except ImportError as e:
                raise ImportError(
                    "pandas is required for return_type='pandas'"
                ) from e
            
            # Flatten 2D measures into one column per dimension
            data_dict = {}
            for f in fields:
                arr = all_data[f]
                if f == "objective":
                    data_dict["objective"] = arr
                elif arr.ndim == 2:
                    for i in range(arr.shape[1]):
                        data_dict[f"{f}_{i}"] = arr[:, i]
            return pd.DataFrame(data_dict)
            
        raise ValueError(f"Unknown return_type: {return_type}")

    def sample_elites(self, n):
        """Samples `n` elites uniformly at random from the archive.

        Args:
            n (int): Number of elites to sample.
        Returns:
            dict: A dict mapping field names to arrays of sampled data.
        Raises:
            IndexError: If the archive is empty.
        """
        if self.empty:
            raise IndexError("No elites in archive to sample.")
            
        # Sample indices with replacement
        indices = self._rng.integers(self._num_elites, size=n)
        
        return {
            "solution": self._solutions[indices],
            "objective": self._objectives[indices],
            "measures": self._measures[indices],
            **{
                name: arr[indices]
                for name, arr in self._extra_field_arrays.items()
            }
        }