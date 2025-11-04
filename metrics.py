
import csv
from typing import Dict, List

class CSVLogger:
    """Logger to save metrics of an experiment in a csv file
    during the training process.
    """

    def __init__(self, filename: str, header: List) -> None:
        """Create the csv logger, create a file and write the
        header.

        Args:
            filename: path to which the file will be saved.
            header: header of the csv file.
        """
        self._filename = filename
        self._header = header
        with open(self._filename, "w") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write the header
            writer.writeheader()

    def log(self, metrics: Dict[str, float]) -> None:
        """Log new metrics to the csv file.

        Args:
            metrics: A dictionary containing the metrics that
                need to be saved.
        """
        with open(self._filename, "a") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write new metrics in a raw
            writer.writerow(metrics)
            

def me_metrics_function(
    repetoire,
    iteration,
    config,
):
    "Calculate MAP-Elites metrics"
    
    current_metrics = {}
    current_metrics["iteration"] = iteration
    current_metrics["evaluations"] = (iteration + 1) * config.emitter.batch_size
    current_metrics["coverage"] = repetoire.stats.coverage
    current_metrics["qd_score"] = repetoire.stats.qd_score
    current_metrics["max_fitness"] = repetoire.stats.obj_max
    
    return current_metrics  