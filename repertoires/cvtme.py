import numbers
import numpy as np
from numpy_groupies import aggregate_nb as aggregate
from scipy.spatial import cKDTree
from scipy.stats.qmc import Halton, Sobol
from sklearn.cluster import k_means

from ribs._utils import (
    check_batch_shape,
    check_finite,
    check_shape,
    validate_batch,
    validate_single,
)
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._array_store import ArrayStore
from ribs.archives._utils import (
    fill_sentinel_values,
    validate_cma_mae_settings,
)

class CVTDictArchive(ArchiveBase):
    """
    A CVT Archive that supports Dictionary-based solutions.
    """

    def __init__(
        self,
        *,
        solution_dim,
        cells,
        ranges,
        learning_rate=None,
        threshold_min=-np.inf,
        qd_score_offset=0.0,
        seed=None,
        dtype=np.float64,
        extra_fields=None,
        custom_centroids=None,
        centroid_method="kmeans",
        samples=100_000,
        k_means_kwargs=None,
        use_kd_tree=True,
        ckdtree_kwargs=None,
        chunk_size=None,
    ):
        self._rng = np.random.default_rng(seed)

        # --- MODIFICATION START: Handle Dict Schema ---
        if isinstance(solution_dim, dict):
            self._solution_schema = solution_dim
            self._is_structured = True
            # Pass empty tuple to Base to bypass scalar checks
            super_dim = ()
        else:
            self._solution_schema = {"solution": solution_dim}
            self._is_structured = False
            super_dim = solution_dim
        # --- MODIFICATION END ---

        ArchiveBase.__init__(
            self,
            solution_dim=super_dim,
            objective_dim=(),
            measure_dim=len(ranges),
        )

        extra_fields = extra_fields or {}
        reserved_fields = {"objective", "measures", "threshold", "index"} # Removed "solution" from reserved
        if reserved_fields & extra_fields.keys():
            raise ValueError(f"Reserved fields: {reserved_fields}")
        
        # --- DTYPE HANDLING ---
        default_dtype = dtype if dtype is not None else np.float64
        
        # Helper to resolve dtype
        def get_dt(key):
            if isinstance(default_dtype, dict):
                return default_dtype.get(key, np.float64)
            return default_dtype
        
        self._dtypes = {
            "solution": get_dt("solution"),
            "objective": get_dt("objective"),
            "measures": get_dt("measures"),
            "threshold": get_dt("objective"),
        }

        # --- MODIFICATION START: Flatten Schema into Field Desc ---
        # Instead of one "solution" field, we create fields for every key in the dict
        # e.g. "latent": (shape, dtype), "prompt": (shape, dtype)
        
        flat_field_desc = {
            "objective": ((), self._dtypes["objective"]),
            "measures": (self.measure_dim, self._dtypes["measures"]),
            "threshold": ((), self._dtypes["objective"]),
            **extra_fields,
        }

        # Add solution components
        for key, shape in self._solution_schema.items():
            # If not structured, key is "solution" which matches original behavior
            flat_field_desc[key] = (shape, self._dtypes["solution"])

        self._store = ArrayStore(
            field_desc=flat_field_desc,
            capacity=cells,
        )
        # --- MODIFICATION END ---

        # ... (Standard CVT Initialization - Unchanged) ...
        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0], dtype=self.dtypes["measures"])
        self._upper_bounds = np.array(ranges[1], dtype=self.dtypes["measures"])
        self._interval_size = self._upper_bounds - self._lower_bounds
        self._learning_rate, self._threshold_min = validate_cma_mae_settings(
            learning_rate, threshold_min, self.dtypes["threshold"]
        )
        self._qd_score_offset = np.asarray(qd_score_offset, dtype=self.dtypes["objective"])

        self._best_elite = None
        self._objective_sum = None
        self._stats = None
        self._stats_reset()

        self._k_means_kwargs = {} if k_means_kwargs is None else k_means_kwargs.copy()
        self._k_means_kwargs.setdefault("n_init", 1)
        self._k_means_kwargs.setdefault("init", "random")
        self._k_means_kwargs.setdefault("algorithm", "lloyd")
        self._k_means_kwargs.setdefault("random_state", seed)

        if custom_centroids is None:
            self._samples = None
            if centroid_method == "kmeans":
                if not isinstance(samples, numbers.Integral):
                    samples = np.asarray(samples, dtype=self.dtypes["measures"])
                    if samples.shape[1] != self._measure_dim:
                        raise ValueError("Sample shape mismatch")
                    self._samples = samples
                else:
                    self._samples = self._rng.uniform(
                        self._lower_bounds,
                        self._upper_bounds,
                        size=(samples, self._measure_dim),
                    ).astype(self.dtypes["measures"])

                self._centroids = k_means(
                    self._samples, self.cells, **self._k_means_kwargs
                )[0]

                if self._centroids.shape[0] < self.cells:
                    raise RuntimeError("K-Means found fewer centroids than cells.")
            elif centroid_method == "random":
                self._centroids = self._rng.uniform(
                    self._lower_bounds,
                    self._upper_bounds,
                    size=(self.cells, self._measure_dim),
                )
            elif centroid_method == "sobol":
                sampler = Sobol(d=self._measure_dim, scramble=False)
                self._centroids = self._lower_bounds + sampler.random(n=self.cells) * self._interval_size
            elif centroid_method == "scrambled_sobol":
                sampler = Sobol(d=self._measure_dim, scramble=True)
                self._centroids = self._lower_bounds + sampler.random(n=self.cells) * self._interval_size
            elif centroid_method == "halton":
                sampler = Halton(d=self._measure_dim)
                self._centroids = self._lower_bounds + sampler.random(n=self.cells) * self._interval_size
        else:
            custom_centroids = np.asarray(custom_centroids, dtype=self.dtypes["measures"])
            if custom_centroids.shape != (cells, self._measure_dim):
                raise ValueError("Custom centroids shape mismatch")
            self._centroids = custom_centroids
            self._samples = None

        self._use_kd_tree = use_kd_tree
        self._centroid_kd_tree = None
        self._ckdtree_kwargs = {} if ckdtree_kwargs is None else ckdtree_kwargs.copy()
        self._chunk_size = chunk_size
        if self._use_kd_tree:
            self._centroid_kd_tree = cKDTree(self._centroids, **self._ckdtree_kwargs)

    # --- HELPER: Repack Dictionary ---
    def _repack_solution(self, flat_data):
        """Helper to restructure flat fields back into a solution dict."""
        if not self._is_structured:
            # Standard behavior: solution is just the "solution" field
            return flat_data
        
        # If structured, extract keys defined in schema
        solution = {}
        for key in self._solution_schema.keys():
            if key in flat_data:
                solution[key] = flat_data.pop(key)
        
        flat_data["solution"] = solution
        return flat_data

    # --- PROPERTIES ---
    
    @property
    def solution_dim(self):
        """Returns the dictionary schema."""
        return self._solution_schema

    @property
    def field_list(self):
        return self._store.field_list_with_index

    @property
    def dtypes(self):
        # [FIX] Return dtypes including "solution" key for compatibility with Emitters
        d = self._store.dtypes_with_index.copy()
        if self._is_structured and "solution" not in d:
            d["solution"] = self._dtypes["solution"]
        return d

    @property
    def stats(self):
        return self._stats

    @property
    def empty(self):
        return len(self._store) == 0
    
    @property
    def best_elite(self):
        return self._best_elite

    @property
    def cells(self):
        return self._store.capacity

    @property
    def centroids(self):
        return self._centroids

    # [FIX] Added missing properties required by add() and other methods
    @property
    def lower_bounds(self):
        return self._lower_bounds

    @property
    def upper_bounds(self):
        return self._upper_bounds

    @property
    def interval_size(self):
        return self._interval_size

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def threshold_min(self):
        return self._threshold_min

    @property
    def qd_score_offset(self):
        return self._qd_score_offset

    @property
    def samples(self):
        return self._samples

    # --- METHODS ---

    def __len__(self):
        return len(self._store)

    def __iter__(self):
        # We need to wrap the iterator to repack dictionaries
        for entry in self._store:
            yield self._repack_solution(entry)

    def _stats_reset(self):
        self._best_elite = None
        dtype = self.dtypes["objective"]
        self._objective_sum = np.asarray(0.0, dtype=dtype)
        self._stats = ArchiveStats(
            num_elites=0,
            coverage=np.asarray(0.0, dtype=dtype),
            qd_score=np.asarray(0.0, dtype=dtype),
            norm_qd_score=np.asarray(0.0, dtype=dtype),
            obj_max=None,
            obj_mean=None,
        )

    def _stats_update(self, new_objective_sum, new_best_index):
        # Retrieve and Repack
        _, new_best_elite = self._store.retrieve([new_best_index])
        
        # Convert single-item arrays to scalars/tensors for the elite dict
        # Also Repack the solution dict logic here
        flat_elite = {k: v[0] for k, v in new_best_elite.items()}
        
        if self._is_structured:
            sol = {}
            for k in self._solution_schema:
                sol[k] = flat_elite.pop(k)
            flat_elite["solution"] = sol
        
        if (self._stats.obj_max is None or flat_elite["objective"] > self._stats.obj_max):
            self._best_elite = flat_elite
            new_obj_max = flat_elite["objective"]
        else:
            new_obj_max = self._stats.obj_max

        self._objective_sum = new_objective_sum
        dtype = self.dtypes["objective"]
        new_qd_score = (self._objective_sum - np.asarray(len(self), dtype=dtype) * self._qd_score_offset)
        
        self._stats = ArchiveStats(
            num_elites=len(self),
            coverage=np.asarray(len(self) / self.cells, dtype=dtype),
            qd_score=new_qd_score,
            norm_qd_score=np.asarray(new_qd_score / self.cells, dtype=dtype),
            obj_max=new_obj_max,
            obj_mean=np.asarray(self._objective_sum / len(self), dtype=dtype),
        )

    def index_of(self, measures):
        """Same as original."""
        measures = np.asarray(measures)
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        if self._use_kd_tree:
            _, indices = self._centroid_kd_tree.query(measures)
            return indices.astype(np.int32)
        else:
            expanded_measures = np.expand_dims(measures, axis=1)
            if self._chunk_size is not None and self._chunk_size < measures.shape[0]:
                indices = []
                chunks = np.array_split(expanded_measures, np.ceil(len(expanded_measures) / self._chunk_size))
                for chunk in chunks:
                    distances = np.sum(np.square(chunk - self.centroids), axis=2)
                    indices.append(np.argmin(distances, axis=1).astype(np.int32))
                return np.concatenate(tuple(indices))
            else:
                distances = np.sum(np.square(expanded_measures - self.centroids), axis=2)
                return np.argmin(distances, axis=1).astype(np.int32)

    def index_of_single(self, measures):
        measures = np.asarray(measures)
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")
        return self.index_of(measures[None])[0]

    @staticmethod
    def _compute_thresholds(indices, objective, cur_threshold, learning_rate, dtype):
        """Same as original."""
        if len(indices) == 0:
            return np.array([], dtype=dtype)
        objective_sizes = aggregate(indices, 1, func="len", fill_value=0)[indices]
        objective_sums = aggregate(indices, objective, func="sum", fill_value=np.nan)[indices]
        ratio = dtype(1.0 - learning_rate) ** objective_sizes
        new_threshold = ratio * cur_threshold + (objective_sums / objective_sizes) * (1 - ratio)
        return new_threshold

    def add(self, solution, objective, measures, **fields):
        """
        Modified add to handle Dictionary Solutions.
        """
        # --- MODIFICATION START: Pre-Validation Unpacking ---
        if self._is_structured:
            if not isinstance(solution, dict):
                raise ValueError("Expected dictionary solution.")
            
            # [FIX] Manually validate dictionary solution to bypass 'solution' key check in validate_batch
            
            # 1. Validate Standard Fields
            objective = np.asarray(objective, dtype=self.dtypes["objective"])
            measures = np.asarray(measures, dtype=self.dtypes["measures"])
            
            batch_size = len(objective)
            check_batch_shape(objective, "objective", (), "objective_dim")
            check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
            check_finite(measures, "measures")
            
            # 2. Start constructing 'data' dictionary with standard fields
            data = {
                "objective": objective,
                "measures": measures,
            }
            
            # 3. Validate and Add Extra Fields
            for k, v in fields.items():
                v = np.asarray(v)
                if v.shape[0] != batch_size:
                     raise ValueError(f"Field '{k}' has batch size {v.shape[0]}, expected {batch_size}")
                data[k] = v

            # 4. Validate and Add Solution Components
            for k, v in solution.items():
                if k not in self._solution_schema:
                     # Skip keys not defined in schema (e.g. they weren't allocated in ArrayStore)
                     continue
                
                v = np.asarray(v, dtype=self.dtypes["solution"])
                expected_shape = self._solution_schema[k]
                check_batch_shape(v, k, expected_shape, "solution_component")
                
                data[k] = v
                
        else:
            # Standard behavior
            data = validate_batch(
                self,
                {
                    "solution": solution,
                    "objective": objective,
                    "measures": measures,
                    **fields,
                },
            )
        # --- MODIFICATION END ---

        del solution, objective, measures, fields

        add_info = {}
        indices = self.index_of(data["measures"])
        batch_size = len(indices)

        cur_occupied, cur_data = self._store.retrieve(indices)
        cur_threshold = cur_data["threshold"]
        cur_threshold[~cur_occupied] = self.threshold_min

        can_insert = data["objective"] > cur_threshold
        is_new = can_insert & ~cur_occupied
        improve_existing = can_insert & cur_occupied
        
        add_info["status"] = np.zeros(batch_size, dtype=np.int32)
        add_info["status"][is_new] = 2
        add_info["status"][improve_existing] = 1

        cur_threshold[is_new] = (
            np.asarray(0.0, dtype=self.dtypes["threshold"])
            if self.threshold_min == -np.inf
            else self.threshold_min
        )
        add_info["value"] = data["objective"] - cur_threshold

        if not np.any(can_insert):
            return add_info

        indices = indices[can_insert]
        data = {name: arr[can_insert] for name, arr in data.items()}
        cur_threshold = cur_threshold[can_insert]

        if self.threshold_min == -np.inf:
            new_threshold = data["objective"]
        else:
            new_threshold = self._compute_thresholds(
                indices,
                data["objective"],
                cur_threshold,
                self.learning_rate,
                self.dtypes["threshold"],
            )

        archive_argmax = aggregate(indices, data["objective"], func="argmax", fill_value=-1)
        should_insert = archive_argmax[archive_argmax != -1]

        indices = indices[should_insert]
        data = {name: arr[should_insert] for name, arr in data.items()}
        data["threshold"] = new_threshold[should_insert]

        self._store.add(indices, data)

        cur_objective = cur_data["objective"]
        cur_objective[~cur_occupied] = 0.0
        cur_objective = cur_objective[can_insert][should_insert]
        objective_sum = self._objective_sum + np.sum(data["objective"] - cur_objective)
        best_index = indices[np.argmax(data["objective"])]
        self._stats_update(objective_sum, best_index)

        return add_info

    def add_single(self, solution, objective, measures, **fields):
        # Wrap the single add logic to support dicts
        # For simplicity in this adaptation, we forward to add() with batch=1
        # This avoids duplicating the messy unpacking logic
        
        # Batch dimensions
        objective_batch = np.expand_dims(objective, axis=0)
        measures_batch = np.expand_dims(measures, axis=0)
        fields_batch = {k: np.expand_dims(v, axis=0) for k, v in fields.items()}
        
        if self._is_structured and isinstance(solution, dict):
            solution_batch = {k: np.expand_dims(v, axis=0) for k, v in solution.items()}
        else:
            solution_batch = np.expand_dims(solution, axis=0)

        info = self.add(solution_batch, objective_batch, measures_batch, **fields_batch)
        return {k: v[0] for k, v in info.items()}

    def clear(self):
        self._store.clear()
        self._stats_reset()

    def retrieve(self, measures):
        measures = np.asarray(measures)
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        occupied, data = self._store.retrieve(self.index_of(measures))
        fill_sentinel_values(occupied, data)

        # --- MODIFICATION: Repack ---
        return occupied, self._repack_solution(data)

    def retrieve_single(self, measures):
        measures = np.asarray(measures)
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        occupied, data = self.retrieve(measures[None])
        
        # If structured, data['solution'] is a dict of batches.
        # We need to extract the 0-th element of every array in that structure
        result_data = {field: arr[0] for field, arr in data.items() if field != "solution"}
        
        if self._is_structured:
            result_data["solution"] = {k: v[0] for k, v in data["solution"].items()}
        else:
            result_data["solution"] = data["solution"][0]

        return occupied[0], result_data

    def data(self, fields=None, return_type="dict"):
        # Get raw flat data from store
        flat_data = self._store.data(fields=None, return_type="dict")
        
        # Repack into structured format
        all_data = self._repack_solution(flat_data)

        # Now filter fields and convert type
        if fields is None:
            fields = self.field_list # Uses store's list, but we need to inject "solution"
            if self._is_structured:
                # Remove individual component keys, add "solution"
                fields = [f for f in fields if f not in self._solution_schema]
                fields.append("solution")
        
        if return_type == "dict":
            return {f: all_data[f] for f in fields}
        
        # Handle other return types logic similar to DNSDictArchive if needed
        # But keeping it minimal for now as per "change as little as possible"
        # The base `_store.data` supports tuple/pandas but on flat data.
        # Here we manually handle dict.
        
        raise ValueError(f"CVTDictArchive currently only fully supports return_type='dict'. Got {return_type}")

    def sample_elites(self, n):
        if self.empty:
            raise IndexError("No elements in archive.")

        random_indices = self._rng.integers(len(self._store), size=n)
        selected_indices = self._store.occupied_list[random_indices]
        _, elites = self._store.retrieve(selected_indices)
        
        return self._repack_solution(elites)