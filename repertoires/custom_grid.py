"""Contains the GridDictArchive."""

from __future__ import annotations

from collections.abc import Collection, Iterator
from typing import Literal, overload, Dict, Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from numpy_groupies import aggregate_nb as aggregate

from ribs._utils import (
    check_batch_shape,
    check_finite,
    check_is_1d,
    check_shape,
    validate_batch,
    validate_single,
)
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._archive_data_frame import ArchiveDataFrame
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._array_store import ArrayStore
from ribs.archives._utils import (
    fill_sentinel_values,
    validate_cma_mae_settings,
)


class GridDictArchive(ArchiveBase):
    """
    A GridArchive that supports dictionary-based solutions via parallel storage.
    
    It maintains the grid logic using PyRibs ArrayStore for metadata (objectives, measures),
    but stores the complex solution dictionary in a separate parallel structure
    to avoid flattening.
    """

    def __init__(
        self,
        *,
        solution_dim: Dict[str, tuple[int, ...]] | int | tuple[int, ...],
        dims: Collection[int],
        ranges: Collection[tuple[float, float]],
        learning_rate: float | None = None,
        threshold_min: float = -np.inf,
        epsilon: float = 1e-6,
        qd_score_offset: float = 0.0,
        seed: int | None = None,
        solution_dtype: DTypeLike = None,
        objective_dtype: DTypeLike = None,
        measures_dtype: DTypeLike = None,
        dtype: DTypeLike = None,
        extra_fields: dict | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._dims = np.array(dims, dtype=np.int32)

        # --- 1. SCHEMA SETUP ---
        if isinstance(solution_dim, dict):
            self._solution_schema = solution_dim
        else:
            shape = (solution_dim,) if isinstance(solution_dim, int) else solution_dim
            self._solution_schema = {"data": shape}

        # Initialize Base with empty dims (we handle storage manually)
        ArchiveBase.__init__(
            self,
            solution_dim=(),
            objective_dim=(),
            measure_dim=len(self._dims),
        )

        extra_fields = extra_fields or {}
        # We allow 'solution' to be absent from ArrayStore, but present in our property
        reserved_fields = {"objective", "measures", "threshold", "index"} 
        if reserved_fields & extra_fields.keys():
            raise ValueError(f"Fields {reserved_fields} are reserved.")

        # --- 2. DTYPE RESOLUTION ---
        default_dtype = dtype if dtype is not None else np.float64
        if isinstance(default_dtype, dict):
            def_s = default_dtype.get("solution", np.float64)
            def_o = default_dtype.get("objective", np.float64)
            def_m = default_dtype.get("measures", np.float64)
        else:
            def_s = def_o = def_m = default_dtype

        self._s_dtype = solution_dtype if solution_dtype is not None else def_s
        o_dtype = objective_dtype if objective_dtype is not None else def_o
        m_dtype = measures_dtype if measures_dtype is not None else def_m

        # --- 3. STORAGE SETUP ---
        
        # A. ArrayStore (Metadata & Grid Logic)
        store_fields = {
            "objective": ((), o_dtype),
            "measures": (self.measure_dim, m_dtype),
            "threshold": ((), o_dtype),
            **extra_fields,
        }
        
        # We add a dummy 'solution' field to ArrayStore so internal logic doesn't break,
        # but it will just store 1-byte placeholders.
        store_fields["solution"] = ((), np.int8) 

        self._store = ArrayStore(
            field_desc=store_fields,
            capacity=np.prod(self._dims),
        )

        # B. Parallel Solution Store (The Real Data)
        # We mirror the grid capacity but use the actual solution schema
        self._solution_store = {}
        for key, shape in self._solution_schema.items():
            self._solution_store[key] = np.empty(
                (self._store.capacity, *shape), 
                dtype=self._s_dtype
            )

        # --- 4. CONSTANTS ---
        if len(self._dims) != len(ranges):
            raise ValueError("dims and ranges length mismatch")
        ranges = list(zip(*ranges, strict=True))
        self._lower_bounds = np.array(ranges[0], dtype=m_dtype)
        self._upper_bounds = np.array(ranges[1], dtype=m_dtype)
        self._interval_size = self._upper_bounds - self._lower_bounds
        self._boundaries = self._compute_boundaries(self._dims, self._lower_bounds, self._upper_bounds)
        self._epsilon = np.asarray(epsilon, dtype=m_dtype)
        self._learning_rate, self._threshold_min = validate_cma_mae_settings(
            learning_rate, threshold_min, o_dtype
        )
        self._qd_score_offset = np.asarray(qd_score_offset, dtype=o_dtype)

        self._best_elite = None
        self._objective_sum = np.asarray(0.0, dtype=o_dtype)
        self._stats = None
        self._stats_reset()

    @staticmethod
    def _compute_boundaries(dims, lower_bounds, upper_bounds):
        boundaries = []
        for dim, lower_bound, upper_bound in zip(dims, lower_bounds, upper_bounds, strict=True):
            boundaries.append(np.linspace(lower_bound, upper_bound, dim + 1))
        return boundaries

    # --- PROPERTIES ---

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        """Override to ensure 'solution' maps to the correct type for Emitters."""
        d = self._store.dtypes_with_index.copy()
        # Report the TRUE solution dtype, not the dummy int8 one
        d["solution"] = self._s_dtype
        return d
    
    @property
    def field_list(self) -> list[str]:
        return self._store.field_list_with_index

    @property
    def stats(self) -> ArchiveStats:
        return self._stats

    @property
    def empty(self) -> bool:
        return len(self._store) == 0
    
    @property
    def best_elite(self) -> Any:
        if self._best_elite is None:
            return None
        # Retrieve full data for best elite
        return self.data(return_type="dict_array")[self.index_of_single(self._best_elite["measures"])]

    @property
    def cells(self) -> int:
        """Total number of cells in the archive."""
        return self._store.capacity

    @property
    def dims(self) -> np.ndarray:
        """(measure_dim,) array listing the number of cells in each dimension."""
        return self._dims

    @property
    def lower_bounds(self) -> np.ndarray:
        """(measure_dim,) array listing the lower bound of each dimension."""
        return self._lower_bounds

    @property
    def upper_bounds(self) -> np.ndarray:
        """(measure_dim,) array listing the upper bound of each dimension."""
        return self._upper_bounds

    @property
    def interval_size(self) -> np.ndarray:
        """(measure_dim,) array listing the size of each dim (upper_bounds - lower_bounds)."""
        return self._interval_size

    @property
    def boundaries(self) -> list[np.ndarray]:
        """The boundaries of the cells in each dimension."""
        return self._boundaries

    @property
    def epsilon(self) -> float:
        """Epsilon for computing archive indices."""
        return self._epsilon

    @property
    def learning_rate(self) -> float:
        """The learning rate for threshold updates."""
        return self._learning_rate

    @property
    def threshold_min(self) -> float:
        """The initial threshold value for all the cells."""
        return self._threshold_min

    @property
    def qd_score_offset(self) -> float:
        """Subtracted from objective values when computing the QD score."""
        return self._qd_score_offset
    
    def add(
        self,
        solution: Dict[str, ArrayLike] | ArrayLike,
        objective: ArrayLike,
        measures: ArrayLike,
        **fields: ArrayLike,
    ) -> BatchData:
        
        # 1. Validation Setup
        objective = np.asarray(objective)
        if objective.ndim == 0: objective = np.expand_dims(objective, axis=0)
        batch_size = len(objective)

        # Helper to ensure (Batch, ...) shape
        def ensure_batch_dim(arr):
            arr = np.asarray(arr)
            if arr.ndim == 0 and batch_size == 1:
                return np.expand_dims(arr, axis=0)
            return arr

        # Fix dimensions
        measures = ensure_batch_dim(measures)
        fields = {k: ensure_batch_dim(v) for k, v in fields.items()}
        
        # Prepare Dummy Solution for validation (so Ribs validate_batch is happy)
        dummy_solution = np.zeros(batch_size, dtype=np.int8)

        # 2. Validate Metadata
        batch_data = {
            "solution": dummy_solution,
            "objective": objective,
            "measures": measures,
            **fields,
        }
        data = validate_batch(self, batch_data)
        
        # 3. Grid Logic (Where do things go?)
        indices = self.index_of(data["measures"])
        
        # Separate metadata from solution
        # NOTE: meta_data should ONLY contain keys that exist in self._store
        valid_store_keys = set(self._store.field_list)
        meta_data = {
            k: v for k, v in data.items() 
            if k in valid_store_keys and k != "solution"
        }

        cur_occupied, cur_data = self._store.retrieve(indices)
        cur_threshold = cur_data["threshold"]
        cur_threshold[~cur_occupied] = self.threshold_min

        can_insert = data["objective"] > cur_threshold
        is_new = can_insert & ~cur_occupied
        improve_existing = can_insert & cur_occupied
        
        add_info = {
            "status": np.zeros(batch_size, dtype=np.int32),
            "value": np.zeros(batch_size, dtype=self.dtypes["objective"])
        }
        add_info["status"][is_new] = 2
        add_info["status"][improve_existing] = 1

        cur_threshold[is_new] = (
            0.0 if self.threshold_min == -np.inf else self.threshold_min
        )
        add_info["value"] = data["objective"] - cur_threshold

        if not np.any(can_insert):
            return add_info

        # 4. Filter for insertion
        indices = indices[can_insert]
        meta_data = {k: v[can_insert] for k, v in meta_data.items()} # Sliced metadata
        cur_threshold = cur_threshold[can_insert]

        # Prepare the REAL solution dict (Merged from 'solution' arg AND 'fields')
        # We need to capture everything that belongs in _solution_store
        
        real_solution = {}
        
        # A. From the explicit 'solution' argument
        if isinstance(solution, dict):
            for k, v in solution.items():
                if k in self._solution_store:
                    real_solution[k] = ensure_batch_dim(v)[can_insert]
        else:
            if "data" in self._solution_store:
                real_solution["data"] = ensure_batch_dim(solution)[can_insert]

        # B. From **fields (kwargs)
        # Often 'image' or descriptors are passed here but belong in solution store
        for k, v in fields.items():
            if k in self._solution_store:
                real_solution[k] = ensure_batch_dim(v)[can_insert]

        # 5. Threshold Updates
        if self.threshold_min == -np.inf:
            # We must use the objective from 'data' because 'meta_data' might not have it 
            # if we filtered strictly (though objective is usually in store).
            new_threshold = data["objective"][can_insert]
        else:
            new_threshold = self._compute_thresholds(
                indices,
                data["objective"][can_insert],
                cur_threshold,
                self.learning_rate,
                self.dtypes["threshold"],
            )

        # 6. Resolve Collisions (Argmax)
        archive_argmax = aggregate(
            indices, data["objective"][can_insert], func="argmax", fill_value=-1
        )
        should_insert = archive_argmax[archive_argmax != -1]

        # Final filtering
        final_indices = indices[should_insert]
        
        # Filter metadata for ArrayStore
        final_meta = {k: v[should_insert] for k, v in meta_data.items()}
        final_meta["threshold"] = new_threshold[should_insert]
        
        # Filter real solution for _solution_store
        final_solution = {k: v[should_insert] for k, v in real_solution.items()}
        
        # 7. WRITE TO STORES
        
        # A. Update ArrayStore (Metadata)
        # We pass a sliced dummy solution to keep it happy
        final_meta["solution"] = np.zeros(len(final_indices), dtype=np.int8)
        
        # CRITICAL FIX: Ensure final_meta ONLY has keys the store expects
        keys_to_add = {k: v for k, v in final_meta.items() if k in valid_store_keys}
        
        self._store.add(final_indices, keys_to_add)

        # B. Update Parallel Solution Store (The Real Data)
        for k, v in final_solution.items():
            if k in self._solution_store:
                 self._solution_store[k][final_indices] = v

        # 8. Update Stats
        # We use the full objective sum from the store for accuracy
        total_sum = np.nansum(self._store._fields["objective"]) 
        
        best_idx_in_batch = np.argmax(final_meta["objective"]) if len(final_meta["objective"]) > 0 else 0
        best_grid_idx = final_indices[best_idx_in_batch] if len(final_indices) > 0 else 0
        
        self._stats_update(total_sum, best_grid_idx)

        return add_info

    def add_single(
        self,
        solution: Dict[str, ArrayLike] | ArrayLike,
        objective: ArrayLike,
        measures: ArrayLike,
        **fields: ArrayLike,
    ) -> Any:
        # Pass to batch add (expand dims)
        objective = np.asarray(objective)
        measures = np.asarray(measures)
        
        sol_batch = {}
        if isinstance(solution, dict):
            for k, v in solution.items():
                sol_batch[k] = np.expand_dims(np.array(v), axis=0)
        else:
            sol_batch = np.expand_dims(np.array(solution), axis=0)

        fields_batch = {k: np.expand_dims(np.array(v), axis=0) for k, v in fields.items()}
        
        batch_info = self.add(
            solution=sol_batch,
            objective=np.expand_dims(objective, axis=0),
            measures=np.expand_dims(measures, axis=0),
            **fields_batch
        )
        
        return {
            "status": batch_info["status"][0],
            "value": batch_info["value"][0],
        }

    def clear(self) -> None:
        """Removes all elites in the archive."""
        self._store.clear()
        self._stats_reset()

    ## READING METHODS ##

    def retrieve(self, measures: ArrayLike) -> tuple[np.ndarray, BatchData]:
        measures = np.asarray(measures, dtype=self.dtypes["measures"])
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        # Get indices
        indices = self.index_of(measures)
        occupied, flat_data = self._store.retrieve(indices)
        fill_sentinel_values(occupied, flat_data)

        # Construct result dict
        data = {k: v for k, v in flat_data.items() if k != "solution"}
        
        # Fetch Real Solutions
        sol_parts = {}
        for k, store_arr in self._solution_store.items():
            # Get valid data, fill unoccupied with nan/zeros
            # We grab everything, then mask. 
            # Note: Unoccupied slots in _solution_store contain garbage/zeros.
            # PyRibs sentinel logic doesn't apply automatically to our custom store.
            retrieved = store_arr[indices]
            sol_parts[k] = retrieved
            
        if "data" in sol_parts and len(sol_parts) == 1:
            data["solution"] = sol_parts["data"]
        else:
            data["solution"] = sol_parts

        return occupied, data

    def retrieve_single(self, measures: ArrayLike) -> tuple[bool, Any]:
        measures = np.asarray(measures, dtype=self.dtypes["measures"])
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        idx = self.index_of_single(measures)
        occupied, flat_data = self._store.retrieve([idx])
        
        # Unpack
        data = {k: v[0] for k, v in flat_data.items() if k != "solution"}
        
        sol_parts = {}
        for k, store_arr in self._solution_store.items():
            sol_parts[k] = store_arr[idx]
            
        if "data" in sol_parts and len(sol_parts) == 1:
            data["solution"] = sol_parts["data"]
        else:
            data["solution"] = sol_parts
        
        return occupied[0], data

    def __len__(self) -> int:
        """Returns the number of elites in the archive."""
        return len(self._store)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterator yielding dictionary elites from the archive."""
        # Iterate over the raw indices of occupied cells
        for idx in self._store.occupied_list:
            # 1. Retrieve metadata from ArrayStore
            # retrieve returns a batch, so we pass [idx] and index [0]
            _, flat_data = self._store.retrieve([idx])
            
            # Unpack metadata, skipping the dummy 'solution' field
            elite = {k: v[0] for k, v in flat_data.items() if k != "solution"}
            
            # 2. Retrieve real solution components from parallel storage
            sol_parts = {}
            for k, store_arr in self._solution_store.items():
                sol_parts[k] = store_arr[idx]
            
            # 3. Structure the solution dictionary
            if "data" in sol_parts and len(sol_parts) == 1:
                elite["solution"] = sol_parts["data"]
            else:
                elite["solution"] = sol_parts
            
            yield elite
            
    @overload
    def data(
        self,
        fields: str,
        return_type: Literal["dict", "tuple", "pandas"] = "dict",
    ) -> np.ndarray: ...

    @overload
    def data(
        self,
        fields: None | Collection[str] = None,
        return_type: Literal["dict"] = "dict",
    ) -> BatchData: ...

    @overload
    def data(
        self,
        fields: None | Collection[str] = None,
        return_type: Literal["tuple"] = "tuple",
    ) -> tuple[np.ndarray]: ...

    @overload
    def data(
        self,
        fields: None | Collection[str] = None,
        return_type: Literal["pandas"] = "pandas",
    ) -> ArchiveDataFrame: ...

    def data(
        self,
        fields: None | Collection[str] | str = None,
        return_type: Literal["dict", "tuple", "pandas"] = "dict",
    ) -> Any:
        
        # Get metadata from ArrayStore
        meta_data = self._store.data(None, "dict")
        # Indices of occupied cells
        occupied_indices = self._store.occupied_list
        
        # Build complete data dict
        data = {k: v for k, v in meta_data.items() if k != "solution"}
        
        # Fetch Real Solutions for occupied cells
        sol_parts = {}
        for k, store_arr in self._solution_store.items():
            sol_parts[k] = store_arr[occupied_indices]
            
        if "data" in sol_parts and len(sol_parts) == 1:
            data["solution"] = sol_parts["data"]
        else:
            data["solution"] = sol_parts

        # Filter fields
        if fields is not None:
            if isinstance(fields, str):
                return data[fields]
            return {k: data[k] for k in fields}

        if return_type == "dict":
            return data
        
        if return_type == "pandas":
            return ArchiveDataFrame(data)
            
        if return_type == "tuple":
            return tuple(data.values())

        return data

    def sample_elites(self, n: int, replace: bool = True) -> BatchData:
        if self.empty:
            raise IndexError("No elements in archive.")
        
        random_indices = self._rng.choice(len(self._store), size=n, replace=replace)
        selected_indices = self._store.occupied_list[random_indices]
        
        # Get metadata
        _, flat_data = self._store.retrieve(selected_indices)
        data = {k: v for k, v in flat_data.items() if k != "solution"}
        
        # Get solutions
        sol_parts = {}
        for k, store_arr in self._solution_store.items():
            sol_parts[k] = store_arr[selected_indices]
        
        if "data" in sol_parts and len(sol_parts) == 1:
            data["solution"] = sol_parts["data"]
        else:
            data["solution"] = sol_parts

        return data

    def retessellate(self, new_dims: Collection[int]) -> None:
        """Updates the resolution of this archive to the given dimensions."""
        if not np.isclose(self.learning_rate, 1):
            raise ValueError("Cannot retessellate an archive with learning rate not equal to 1.")
        if len(new_dims) != self.measure_dim:
            raise ValueError("Dim mismatch")

        # Get all current data
        cur_data = self.data()
        if "index" in cur_data:
            del cur_data["index"]

        self._dims = np.array(new_dims, dtype=np.int32)
        self._boundaries = self._compute_boundaries(
            self._dims, self._lower_bounds, self._upper_bounds
        )
        
        # Resize Stores
        new_capacity = np.prod(self._dims)
        self._store = ArrayStore(self._store.field_desc, capacity=new_capacity)
        
        self._solution_store = {}
        for key, shape in self._solution_schema.items():
            self._solution_store[key] = np.empty(
                (new_capacity, *shape), 
                dtype=self._s_dtype
            )
        
        # Add back data
        self.add(**cur_data)
    
    def _stats_update(self, new_objective_sum: float, new_best_index: float) -> None:
        """Updates statistics."""
        _, new_best_elite = self._store.retrieve([new_best_index])
        new_best_elite = {k: v[0] for k, v in new_best_elite.items()}

        if (
            self._stats.obj_max is None
            or new_best_elite["objective"] > self._stats.obj_max
        ):
            self._best_elite = new_best_elite
            new_obj_max = new_best_elite["objective"]
        else:
            new_obj_max = self._stats.obj_max

        self._objective_sum = new_objective_sum
        
        new_qd_score = (
            self._objective_sum
            - np.asarray(len(self), dtype=self.dtypes["objective"])
            * self._qd_score_offset
        )
        
        self._stats = ArchiveStats(
            num_elites=len(self),
            coverage=np.asarray(len(self) / self.cells, dtype=self.dtypes["objective"]),
            qd_score=new_qd_score,
            norm_qd_score=np.asarray(
                new_qd_score / self.cells, dtype=self.dtypes["objective"]
            ),
            obj_max=new_obj_max,
            obj_mean=np.asarray(
                self._objective_sum / len(self), dtype=self.dtypes["objective"]
            ),
        )

    # --- GRID HELPERS ---
    def index_of(self, measures: ArrayLike) -> np.ndarray:
        measures = np.asarray(measures, dtype=self.dtypes["measures"])
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")
        grid_indices = ((self._dims * (measures - self._lower_bounds) + self._epsilon) / self._interval_size).astype(np.int32)
        grid_indices = np.clip(grid_indices, 0, self._dims - 1)
        return self.grid_to_int_index(grid_indices)

    def index_of_single(self, measures: ArrayLike) -> int:
        measures = np.asarray(measures, dtype=self.dtypes["measures"])
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")
        return self.index_of(measures[None])[0]

    def grid_to_int_index(self, grid_indices: ArrayLike) -> np.ndarray:
        grid_indices = np.asarray(grid_indices)
        check_batch_shape(grid_indices, "grid_indices", self.measure_dim, "measure_dim")
        return np.ravel_multi_index(grid_indices.T, self._dims).astype(np.int32)

    def int_to_grid_index(self, int_indices: ArrayLike) -> np.ndarray:
        int_indices = np.asarray(int_indices)
        check_is_1d(int_indices, "int_indices")
        return np.asarray(np.unravel_index(int_indices, self._dims), dtype=np.int32).T
    
    def _stats_update(self, new_objective_sum: float, new_best_index: float) -> None:
        """Updates statistics.

        Update is based on a new sum of objective values (new_objective_sum) and the
        index of a potential new best elite (new_best_index).
        """
        # Retrieve the best elite at the given index to check if it's the global best
        _, new_best_elite = self._store.retrieve([new_best_index])
        
        # Helper to unpack single-element arrays from store
        new_best_elite = {k: v[0] for k, v in new_best_elite.items()}

        if (
            self._stats.obj_max is None
            or new_best_elite["objective"] > self._stats.obj_max
        ):
            self._best_elite = new_best_elite
            new_obj_max = new_best_elite["objective"]
        else:
            new_obj_max = self._stats.obj_max

        self._objective_sum = new_objective_sum
        
        # Calculate QD Score (Sum of objectives - offset)
        new_qd_score = (
            self._objective_sum
            - np.asarray(len(self), dtype=self.dtypes["objective"])
            * self._qd_score_offset
        )
        
        # Update the stats object
        self._stats = ArchiveStats(
            num_elites=len(self),
            coverage=np.asarray(len(self) / self.cells, dtype=self.dtypes["objective"]),
            qd_score=new_qd_score,
            norm_qd_score=np.asarray(
                new_qd_score / self.cells, dtype=self.dtypes["objective"]
            ),
            obj_max=new_obj_max,
            obj_mean=np.asarray(
                self._objective_sum / len(self), dtype=self.dtypes["objective"]
            ),
        )

    def _stats_reset(self) -> None:
        """Resets the archive stats."""
        self._best_elite = None
        # Use o_dtype from dtypes if available, else float64
        dtype = self.dtypes.get("objective", np.float64)
        
        self._objective_sum = np.asarray(0.0, dtype=dtype)
        self._stats = ArchiveStats(
            num_elites=0,
            coverage=np.asarray(0.0, dtype=dtype),
            qd_score=np.asarray(0.0, dtype=dtype),
            norm_qd_score=np.asarray(0.0, dtype=dtype),
            obj_max=None,
            obj_mean=None,
        )
        
