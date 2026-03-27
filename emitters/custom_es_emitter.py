"""Provides the DictEvolutionStrategyEmitter."""

from __future__ import annotations

import numbers
from collections.abc import Callable, Collection
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ribs.archives import ArchiveBase
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters.opt import EvolutionStrategyBase
from ribs.emitters.rankers import RankerBase, _get_ranker
from ribs.typing import BatchData, Float, Int

from emitters.custom_sep_cma_es import SeparableCMAESDict

# NOTE: Ensure your custom dict-aware optimisers are imported
# from your_custom_module import SeparableCMAESDict, CMAESDict


class DictEvolutionStrategyEmitter(EmitterBase):
    """Adapts a distribution of dictionary solutions with an ES.

    This emitter originates in Fontaine 2020. It has been modified to support
    dictionary-based solutions and handles the routing for dictionary-aware
    optimisers.

    Args:
        archive: Archive of solutions, e.g., ribs.archives.GridArchive.
        initial_solution: Initial dictionary solution.
        mutation_key: The string key in the dictionary that the ES will mutate.
        sigma0: Initial step size / standard deviation of the distribution.
        ranker: The ranker is a RankerBase object that orders the solutions. 
            May be a string abbreviation (e.g. "imp", "2imp").
        es: The evolution strategy class or string. Accepts "cma_es" or 
            "sep_cma_es" to automatically load the dictionary-aware versions, 
            or a direct callable class.
        es_kwargs: Additional arguments to pass to the evolution strategy optimiser.
        selection_rule: Method for selecting parents. With "mu", the first half 
            of the solutions will be selected. With "filter", any solutions added 
            to the archive will be selected.
        restart_rule: Method to use when checking for restarts. If an integer,
            restarts after this many iterations. With "basic", standard CMA-ES
            rules are used. With "no_improvement", restarts when no solutions 
            were added.
        bounds: Bounds of the solution space. Pass None for no bounds.
        lower_bounds: Array specifying the lower bounds.
        upper_bounds: Array specifying the upper bounds.
        batch_size: Number of solutions to return in `ask`.
        seed: Value to seed the random number generator.

    Raises:
        ValueError: If `restart_rule`, `selection_rule`, or `ranker` is invalid.
    """

    def __init__(
        self,
        archive: ArchiveBase,
        *,
        initial_solution: dict,
        mutation_key: str,
        sigma0: Float,
        ranker: Callable[[Int | None], RankerBase] | str = "2imp",
        es: Callable[..., EvolutionStrategyBase] | str = "sep_cma_es",
        es_kwargs: dict | None = None,
        selection_rule: Literal["mu", "filter"] = "filter",
        restart_rule: Literal["no_improvement", "basic"] | int = "no_improvement",
        bounds: Collection[tuple[None | Float, None | Float]] | None = None,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
        batch_size: Int | None = None,
        seed: Int | None = None,
    ) -> None:
        EmitterBase.__init__(
            self,
            archive,
            solution_dim=(),
            bounds=bounds,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        seed_sequence = (
            seed
            if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )
        opt_seed, ranker_seed = seed_sequence.spawn(2)

        self._initial_solution = initial_solution
        self.mutation_key = mutation_key
        self._sigma0 = sigma0

        if selection_rule not in ["mu", "filter"]:
            raise ValueError(f"Invalid selection_rule {selection_rule}")
        self._selection_rule = selection_rule

        self._restart_rule = restart_rule
        self._restarts = 0
        self._itrs = 0

        _ = self._check_restart(0)

        # Intercept string inputs to map to our custom dictionary-aware ES classes
        if isinstance(es, str):
            if es == "sep_cma_es":
                es_class = SeparableCMAESDict
            elif es == "cma_es":
                # Assuming you wrap the standard CMAEvolutionStrategy similarly
                # NOT IMPLEMENTED HERE - you would need to create a CMAESDict class that handles dict solutions
                raise NotImplementedError("Standard CMA-ES for dict solutions is not implemented. Please use 'sep_cma_es' or provide a custom class.")
            else:
                raise ValueError(f"Unsupported ES string for Dict Emitter: {es}")
        else:
            # If a class reference is passed directly, use it
            es_class = es

        self._opt = es_class(
            initial_solution=self._initial_solution,
            mutation_key=self.mutation_key,
            sigma0=sigma0,
            batch_size=batch_size,
            seed=opt_seed,
            lower_bounds=self.lower_bounds if self.lower_bounds is not None else -np.inf,
            upper_bounds=self.upper_bounds if self.upper_bounds is not None else np.inf,
            **(es_kwargs if es_kwargs is not None else {}),
        )
        self._opt.reset(self._initial_solution)

        self._ranker = _get_ranker(ranker, ranker_seed)
        self._ranker.reset(self, archive)

        self._batch_size = self._opt.batch_size

    @property
    def initial_solution(self) -> dict:
        """Initial dictionary solution for the optimiser."""
        return self._initial_solution

    @property
    def batch_size(self) -> Int:
        """Number of solutions to return in ask."""
        return self._batch_size

    @property
    def restarts(self) -> int:
        """The number of restarts for this emitter."""
        return self._restarts

    @property
    def itrs(self) -> int:
        """The number of iterations for this emitter."""
        return self._itrs

    def ask(self) -> dict:
        """Samples new solutions from a multivariate Gaussian.

        Returns:
            A dictionary containing batches of parameters.
        """
        return self._opt.ask()

    def _check_restart(self, num_parents: int) -> bool:
        """Emitter-side checks for restarting the optimiser."""
        if isinstance(self._restart_rule, numbers.Integral):
            return self._itrs % self._restart_rule == 0
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        if self._restart_rule == "basic":
            return False
        raise ValueError(f"Invalid restart_rule {self._restart_rule}")

    def tell(
        self,
        solution: dict,
        objective: ArrayLike,
        measures: ArrayLike,
        add_info: BatchData,
        **fields: ArrayLike,
    ) -> None:
        """Gives the emitter results from evaluating solutions."""
        # Standard validate_batch requires arrays, so we build the data dict manually
        data = {
            "solution": solution,
            "objective": np.asarray(objective),
            "measures": np.asarray(measures),
            **{k: np.asarray(v) for k, v in fields.items()}
        }

        self._itrs += 1

        new_sols = add_info["status"].astype(bool).sum()

        indices, ranking_values = self._ranker.rank(self, self.archive, data, add_info)

        num_parents = (
            new_sols if self._selection_rule == "filter" else self._batch_size // 2
        )

        self._opt.tell(indices, ranking_values, num_parents)

        if self._opt.check_stop(ranking_values[indices]) or self._check_restart(
            new_sols
        ):
            elites = self.archive.sample_elites(1)
            
            if isinstance(elites["solution"], dict):
                new_x0 = {k: v[0] for k, v in elites["solution"].items()}
            else:
                new_x0 = elites["solution"][0]

            self._opt.reset(new_x0)
            self._ranker.reset(self, self.archive)
            self._restarts += 1