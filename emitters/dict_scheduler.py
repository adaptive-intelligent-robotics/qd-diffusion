import numpy as np
import torch  
from ribs.schedulers import Scheduler

class DictScheduler(Scheduler):
    """
    A Scheduler that supports Dictionary-based solutions.
    """
    
    def __init__(self, archive, emitters, result_archive=None, add_mode="batch"):
        super().__init__(archive, emitters, result_archive=result_archive, add_mode=add_mode)
        if not hasattr(self, "_solution_batch_sizes"):
            self._solution_batch_sizes = [e.batch_size for e in self._emitters]

    def ask(self):
        self._cur_solutions = []
        for emitter in self._emitters:
            self._cur_solutions.append(emitter.ask())

        if len(self._cur_solutions) == 0:
            return None

        if not isinstance(self._cur_solutions[0], dict):
            return np.concatenate(self._cur_solutions, axis=0)

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
        Updates the archive.
        
        Logic Update:
        If 'fields' contains keys that exist in the archive's solution_dim (e.g., 'image'),
        we merge them into the solution dictionary before calling archive.add().
        """
        if self._cur_solutions is None:
            raise RuntimeError("ask() must be called before tell().")

        if objective.ndim == 0:
            objective = np.array([objective])
        
        total_size = sum(self._solution_batch_sizes)
        if len(objective) != total_size:
            raise ValueError(f"Objective length {len(objective)} != total batch size {total_size}")

        
        cpu_safe_fields = {}
        for k, v in fields.items():
            if isinstance(v, torch.Tensor):
                cpu_safe_fields[k] = v.detach().cpu().numpy()
            elif isinstance(v, list):
                cpu_safe_fields[k] = np.array(v)
            else:
                cpu_safe_fields[k] = v
        
        # Use sanitized_fields from here on
        fields = cpu_safe_fields
        
        # 1. Reconstruct Latents/Embeddings from 'ask'
        full_solution = {}
        if isinstance(self._cur_solutions[0], dict):
            keys = self._cur_solutions[0].keys()
            for key in keys:
                full_solution[key] = np.concatenate(
                    [batch[key] for batch in self._cur_solutions], axis=0
                )
        else:
            # Fallback for non-dict
            full_solution = np.concatenate(self._cur_solutions, axis=0)

        # 2. [NEW] Merge "Phenotype" (Images) into Solution
        # We look for keys in 'fields' that are actually part of the solution schema
        fields_to_keep = {}
        
        # Check what keys the archive expects in the solution
        if hasattr(self.archive, "solution_dim") and isinstance(self.archive.solution_dim, dict):
            solution_keys = self.archive.solution_dim.keys()
            
            for k, v in fields.items():
                if k in solution_keys:
                    # It is part of the solution (e.g., "image")
                    full_solution[k] = v
                else:
                    # It is strictly metadata (e.g., extra info)
                    fields_to_keep[k] = v
        else:
            fields_to_keep = fields

        # 3. Add to Archive (Now full_solution has latents + images)
        add_info = {}
        if self.archive is not None:
            # Pass cleaned fields (without 'image') as kwargs
            add_info = self.archive.add(full_solution, objective, measures, **fields_to_keep)
            
        if self.result_archive is not None:
            self.result_archive.add(full_solution, objective, measures, **fields_to_keep)

        # 4. Distribute to Emitters
        # (Emitters usually don't need the image, just the fitness/latents)
        cur_idx = 0
        for i, emitter in enumerate(self._emitters):
            batch_size = self._solution_batch_sizes[i]
            
            obj_batch = objective[cur_idx : cur_idx + batch_size]
            meas_batch = measures[cur_idx : cur_idx + batch_size]
            
            # Slice the remaining metadata fields
            fields_batch = {
                k: v[cur_idx : cur_idx + batch_size] for k, v in fields_to_keep.items()
            }

            emitter_add_info = {
                k: v[cur_idx : cur_idx + batch_size] 
                for k, v in add_info.items()
            }

            sol_batch = self._cur_solutions[i]

            emitter.tell(
                sol_batch, 
                obj_batch, 
                meas_batch, 
                emitter_add_info, 
                **fields_batch
            )

            cur_idx += batch_size

        self._cur_solutions = None