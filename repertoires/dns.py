import numpy as np

from ribs._utils import (
    validate_batch,
)
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._utils import (
    parse_dtype,
) 

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

    ##
    ## Core Methods
    ##

    def add(self, solution, objective, measures, **fields):
        """Adds a batch of solutions to the archive.

        This method implements the core DNS logic:
        1. Concatenate new solutions with existing elites.
        2. Calculate the "fitter" matrix.
        3. Calculate distances between all solutions.
        4. Compute the DNS score (mean dist to k-fittest-neighbors).
        5. Sort all solutions by this score.
        6. Keep only the top `max_size` solutions.
        
        Note: This operation can be slow in NumPy for large archives,
        as it involves computing an (N+M) x (N+M) distance matrix,
        where N is `num_elites` and M is `batch_size`.
        """
        
        # 1. Validation
        # Use the pyribs utility to validate and standardize inputs
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
        
        # Information to return
        add_info = {
            "status": np.zeros(batch_size, dtype=np.int32),
            "value": np.full(batch_size, np.nan, dtype=self.dtypes["objective"]),
        }

        # 2. Concatenate
        cur_elites = self._num_elites
        
        # Combine current and new data
        all_sols = np.concatenate(
            (self._solutions[:cur_elites], data["solution"])
        )
        all_objs = np.concatenate(
            (self._objectives[:cur_elites], data["objective"])
        )
        all_measures = np.concatenate(
            (self._measures[:cur_elites], data["measures"])
        )
        
        all_extra_fields = {}
        for name in self._extra_fields:
            all_extra_fields[name] = np.concatenate(
                (self._extra_field_arrays[name][:cur_elites], data[name])
            )
        
        num_all = len(all_objs)
        if num_all == 0:
            return add_info # Nothing to add

        # 3. Calculate "fitter" matrix
        # fitter[i, j] is True if j is fitter than i
        fitter = all_objs[:, None] < all_objs[None, :]
        # An individual cannot be fitter than itself
        np.fill_diagonal(fitter, False)

        # 4. Calculate distances
        if self._distance_metric == "euclidean":
            # Use broadcasting to create a diff matrix
            diff = all_measures[:, None, :] - all_measures[None, :, :]
            # Shape: (num_all, num_all)
            distance = np.linalg.norm(diff, axis=-1)
        
        elif self._distance_metric == "cosine":
            norm = np.linalg.norm(all_measures, axis=1)
            # Handle potential division by zero if a measure vector is all zeros
            norm[norm == 0] = 1e-6
            dot = all_measures @ all_measures.T
            # Shape: (num_all, num_all)
            distance = 1.0 - dot / (norm[:, None] * norm[None, :])
            # Clip to avoid floating point errors
            distance = np.clip(distance, 0.0, 2.0)

        # Mask distances: only consider distances to fitter individuals
        distance = np.where(fitter, distance, np.inf)

        # 5. Compute DNS score (mean dist to k-fittest-neighbors)
        
        # Use partition to find the k smallest distances (fast)
        # k_nearest_indices shape: (num_all, k)
        k_nearest_indices = np.argpartition(
            distance, self._k, axis=1
        )[:, :self._k]
        
        # k_nearest_dists shape: (num_all, k)
        k_nearest_dists = np.take_along_axis(
            distance, k_nearest_indices, axis=1
        )
        
        # Compute mean, ignoring infs (which mean < k fitter neighbors)
        # Replace inf with nan to use nanmean
        k_nearest_dists[np.isinf(k_nearest_dists)] = np.nan
        
        # dns_score shape: (num_all,)
        dns_score = np.nanmean(k_nearest_dists, axis=1)
        
        # If all k neighbors were inf, nanmean returns nan.
        # This means no fitter neighbors were found. Set score to inf.
        dns_score[np.isnan(dns_score)] = np.inf
        
        # 6. Sort and Truncate
        
        # Sort by score descending (higher score is better)
        sorted_indices = np.argsort(dns_score)[::-1]
        
        # Keep only the top max_size
        keep_indices = sorted_indices[:self._max_size]
        new_num_elites = len(keep_indices)

        # 7. Calculate add_info (status)
        
        # Find which *new* solutions survived the cull
        # Indices of new solutions in the `all_` arrays
        new_sol_indices_in_all = np.arange(cur_elites, num_all)
        
        # Check which of the new solutions are in the `keep_indices`
        # `survived` is a boolean array of shape (batch_size,)
        survived = np.isin(new_sol_indices_in_all, keep_indices)
        
        # Status 2: new elite added. Status 0: not added.
        add_info["status"] = np.where(survived, 2, 0).astype(np.int32)
        # Value is just the objective (as in pyribs GridArchive for status=2)
        add_info["value"] = data["objective"]

        # 8. Update Archive State
        self._solutions[:new_num_elites] = all_sols[keep_indices]
        self._objectives[:new_num_elites] = all_objs[keep_indices]
        self._measures[:new_num_elites] = all_measures[keep_indices]
        
        for name in self._extra_fields:
            self._extra_field_arrays[name][:new_num_elites] = \
                all_extra_fields[name][keep_indices]
        
        self._num_elites = new_num_elites
        
        # Fill the rest of the buffer with -inf
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
            
            # This part is complex to generalize for all field shapes.
            # For simplicity, we'll flatten solutions and measures.
            data_dict = {}
            for f in fields:
                arr = all_data[f]
                if f == "objective":
                    data_dict["objective"] = arr
                elif arr.ndim == 2:
                    for i in range(arr.shape[1]):
                        data_dict[f"{f}_{i}"] = arr[:, i]
                # Add more logic here if solutions are multi-dimensional
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