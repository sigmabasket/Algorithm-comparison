import math
import statistics
import tracemalloc
from typing import Callable, Tuple, Dict, List


class MemoryMeasurer:
    def __init__(self):
        self.t_table: Dict[int, float] = {
            6: 2.5706,
            11: 2.2281,
            16: 2.1314,
            21: 2.0860,
            26: 2.0555,
            31: 2.0423,
            36: 2.0301,
            41: 2.0211,
            51: 2.0086,
            101: 1.9840
        }

    def _get_t_value(self, n: int) -> float:
        valid_keys = [k for k in self.t_table if k <= n]
        if not valid_keys:
            raise ValueError("Слишком малое количество запусков. Минимум: 6")
        return self.t_table[max(valid_keys)]

    def measure(
        self,
        func: Callable,
        args: Tuple,
        n_runs: int = 101
    ) -> Tuple[float, float]:
        if n_runs < 6:
            raise ValueError("Количество запусков должно быть не менее 6.")

        usages: List[int] = []

        for _ in range(n_runs):
            tracemalloc.start()
            func(*args)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            usages.append(peak)

        mean_usage = statistics.mean(usages)
        std_dev = statistics.stdev(usages)
        t_value = self._get_t_value(n_runs)
        delta = t_value * (std_dev / math.sqrt(n_runs))

        return mean_usage, delta
