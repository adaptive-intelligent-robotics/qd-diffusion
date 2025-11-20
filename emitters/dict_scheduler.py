import numpy as np
from ribs.schedulers import Scheduler

class DictScheduler(Scheduler):
    """
    A Scheduler that supports Dictionary-based solutions.
    """
    
    def __init__(self, archive, emitters, result_archive=None, add_mode="batch"):
        super().__init__(archive, emitters, result_archive=result_archive, add_mode=add_mode)
        # Safety fallback for batch sizes
        if not hasattr(self, "_solution_batch_sizes"):
            self._solution_batch_sizes = [e.batch_size for e in self._emitters]

    def ask(self):
        self._cur_solutions = []
        for emitter in self._emitters:
            self._cur_solutions.append(emitter.ask())

        if len(self._cur_solutions) == 0:
            return None

        # Standard Array fallback
        if not isinstance(self._cur_solutions[0], dict):
            return np.concatenate(self._cur_solutions, axis=0)

        # Dictionary Concatenation
        concatenated_sol = {}
        keys = self._cur_solutions[0].keys()
        for key in keys:
            concatenated_sol[key] = np.concatenate(
                [batch[key] for batch in self._cur_solutions], 
                axis=0
            )
        return concatenated_sol

    def tell(self, objective, measures, **fields):
        """
        Updates the archive and distributes results (including add_info) to emitters.
        """
        if self._cur_solutions is None:
            raise RuntimeError("ask() must be called before tell().")

        total_size = sum(self._solution_batch_sizes)
        if len(objective) != total_size:
            raise ValueError(f"Objective length {len(objective)} != total batch size {total_size}")

        # 1. Reconstruct Full Solution Dict
        full_solution = {}
        if isinstance(self._cur_solutions[0], dict):
            keys = self._cur_solutions[0].keys()
            for key in keys:
                full_solution[key] = np.concatenate(
                    [batch[key] for batch in self._cur_solutions], axis=0
                )
        else:
            full_solution = np.concatenate(self._cur_solutions, axis=0)

        # 2. Add to Archive and CAPTURE add_info
        add_info = {}
        if self.archive is not None:
            # [FIX] Capture the return value here!
            add_info = self.archive.add(full_solution, objective, measures, **fields)

        # 3. Distribute to Emitters
        cur_idx = 0
        for i, emitter in enumerate(self._emitters):
            batch_size = self._solution_batch_sizes[i]
            
            # Slice standard data
            obj_batch = objective[cur_idx : cur_idx + batch_size]
            meas_batch = measures[cur_idx : cur_idx + batch_size]
            fields_batch = {
                k: v[cur_idx : cur_idx + batch_size] for k, v in fields.items()
            }

            # [FIX] Slice the add_info dictionary for this specific emitter
            # add_info contains arrays like 'status' and 'value' that need splitting
            emitter_add_info = {
                k: v[cur_idx : cur_idx + batch_size] 
                for k, v in add_info.items()
            }

            # Retrieve the specific solution batch
            sol_batch = self._cur_solutions[i]

            # Pass everything to emitter
            emitter.tell(
                sol_batch, 
                obj_batch, 
                meas_batch, 
                emitter_add_info, # <--- Passed as positional arg
                **fields_batch
            )

            cur_idx += batch_size

        self._cur_solutions = None