import math
import statistics
import multiprocessing
from typing import Callable, Tuple, Dict


class TimeMeasurer:
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
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
        """Возвращает t-значение для заданного n
        (ближайшее меньшее из таблицы)."""
        valid_keys = [k for k in self.t_table if k <= n]
        if not valid_keys:
            raise ValueError("Слишком малое количество запусков. Минимум: 6")
        return self.t_table[max(valid_keys)]

    def _timed_run(self, func: Callable, args: Tuple, return_dict):
        import time
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        return_dict["time"] = end - start

    def measure(
        self,
        func: Callable,
        args: Tuple,
        n_runs: int = 101
    ) -> Tuple[float, float]:
        if n_runs < 6:
            raise ValueError("Количество запусков должно быть не менее 6.")

        times = []
        for _ in range(n_runs):
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            proc = multiprocessing.Process(
                target=self._timed_run,
                args=(
                    func,
                    args,
                    return_dict
                )
            )
            proc.start()
            proc.join(timeout=self.timeout)

            if proc.is_alive():
                proc.terminate()
                raise TimeoutError(
                    f"Функция превысила таймаут в {self.timeout} секунд."
                )

            times.append(return_dict["time"])

        mean_time = statistics.mean(times)
        std_dev = statistics.stdev(times)
        t_value = self._get_t_value(n_runs)
        delta = t_value * (std_dev / math.sqrt(n_runs))

        return mean_time, delta
