
import numpy as np
from ribs.emitters import GaussianEmitter

class CustomGaussianEmitter(GaussianEmitter):
    """
    Custom GaussianEmitter that supports multi-dimensional solution spaces
    (i.e., solution_dim is a tuple).
    
    This is fixed by changing this one line in the ask() method:
    Original: size=(self.batch_size, self.solution_dim)
    Fixed:    size=(self.batch_size,) + self.solution_dim
    """
    def ask(self):
        """Creates solutions by adding Gaussian noise to elites in the archive.
        
        (This is copied from the pyribs source, with one line changed)
        """
        if self.archive.empty and self.initial_solutions is not None:
            return self._clip(self.initial_solutions)

        if self.archive.empty:
            parents = np.repeat(self.x0[None], repeats=self.batch_size, axis=0)
        else:
            parents = self.archive.sample_elites(self.batch_size)["solution"]

        noise_shape = (self.batch_size,) + self.solution_dim

        noise = self._rng.normal(
            scale=self.sigma,
            size=noise_shape,  # Use the corrected shape
        ).astype(self.archive.dtypes["solution"])

        return self._clip(parents + noise)