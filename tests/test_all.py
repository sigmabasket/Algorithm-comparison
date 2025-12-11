import os
import unittest
import json
from unittest.mock import patch
from benchmark.time_measurer import TimeMeasurer
from benchmark.memory_measurer import MemoryMeasurer
from src.data_generator import TestDataGenerator
from src.algorithms import (
    naive_search, kmp_search, boyer_moore_search, rabin_karp_search,
    aho_corasick_search, apostolico_crochemore_search
)
from benchmark import time_measurer
import benchmark


class TestSubstringSearchAlgorithms(unittest.TestCase):
    def setUp(self):
        self.algorithms = [
            naive_search,
            kmp_search,
            boyer_moore_search,
            rabin_karp_search
        ]

    def test_exact_match(self):
        text = "abcde"
        pattern = "cde"
        for algo in self.algorithms:
            with self.subTest(algorithm=algo.__name__):
                self.assertEqual(algo(text, pattern), 2)

    def test_no_match(self):
        text = "abcdefgh"
        pattern = "xyz"
        for algo in self.algorithms:
            with self.subTest(algorithm=algo.__name__):
                self.assertEqual(algo(text, pattern), -1)

    def test_match_at_start(self):
        text = "abcdef"
        pattern = "abc"
        for algo in self.algorithms:
            with self.subTest(algorithm=algo.__name__):
                self.assertEqual(algo(text, pattern), 0)

    def test_match_at_end(self):
        text = "abcdef"
        pattern = "def"
        for algo in self.algorithms:
            with self.subTest(algorithm=algo.__name__):
                self.assertEqual(algo(text, pattern), 3)

    def test_full_match(self):
        text = "pattern"
        pattern = "pattern"
        for algo in self.algorithms:
            with self.subTest(algorithm=algo.__name__):
                self.assertEqual(algo(text, pattern), 0)

    def test_empty_pattern(self):
        text = "abc"
        pattern = ""
        for algo in self.algorithms:
            with self.subTest(algorithm=algo.__name__):
                self.assertEqual(algo(text, pattern), 0)

    def test_empty_text(self):
        text = ""
        pattern = "abc"
        for algo in self.algorithms:
            with self.subTest(algorithm=algo.__name__):
                self.assertEqual(algo(text, pattern), -1)

    def test_both_empty(self):
        text = ""
        pattern = ""
        for algo in self.algorithms:
            with self.subTest(algorithm=algo.__name__):
                self.assertEqual(algo(text, pattern), 0)

    def test_repeated_pattern(self):
        text = "ababababab"
        pattern = "abab"
        for algo in self.algorithms:
            with self.subTest(algorithm=algo.__name__):
                self.assertEqual(algo(text, pattern), 0)


class TestTimeMeasurer(unittest.TestCase):
    def test_measure_returns_positive_time_and_delta(self):
        measurer = TimeMeasurer()
        text = "A" * 1000
        pattern = "A" * 5
        time, delta = measurer.measure(
            naive_search, (text, pattern), n_runs=10
        )
        self.assertGreaterEqual(time, 0)
        self.assertGreaterEqual(delta, 0)


class TestMemoryMeasurer(unittest.TestCase):
    def test_measure_returns_positive_memory_and_delta(self):
        measurer = MemoryMeasurer()
        text = "A" * 1000
        pattern = "A" * 5
        memory, delta = measurer.measure(
            naive_search, (text, pattern), n_runs=10
        )
        self.assertGreater(memory, 0)
        self.assertGreaterEqual(delta, 0)


class TestTestDataGenerator(unittest.TestCase):
    def test_generate_all_cases_structure(self):
        generator = TestDataGenerator()
        data = generator.generate_all_cases()
        self.assertIn("naive", data)
        self.assertIn("best", data["naive"])
        self.assertIsInstance(data["naive"]["best"], list)
        self.assertIsInstance(data["naive"]["best"][0], tuple)
        self.assertEqual(len(data["naive"]["best"][0]), 2)


class TestTimeMeasurerExtras(unittest.TestCase):
    def test_get_t_value_known(self):
        tm = time_measurer.TimeMeasurer()
        self.assertAlmostEqual(tm._get_t_value(6), 2.5706, delta=0.01)

    def test_get_t_value_min(self):
        tm = time_measurer.TimeMeasurer()
        with self.assertRaises(ValueError):
            tm._get_t_value(1)


class TestBenchmarkStructure(unittest.TestCase):
    def test_adaptive_n_runs_boundaries(self):
        self.assertGreaterEqual(benchmark.get_adaptive_n_runs(1024), 10)
        self.assertLessEqual(benchmark.get_adaptive_n_runs(2**24), 101)

    def test_save_results_creates_json(self):
        time_data = {
            "naive": {"best": [{"size": 1024, "time": 0.001, "delta": 0.0001}]}
        }
        mem_data = {
            "naive": {"best": [{"size": 1024, "memory": 100, "delta": 1}]}
        }
        benchmark.save_results(time_data, mem_data)
        with open("time_results.json", encoding="utf-8") as f:
            time_loaded = json.load(f)
        with open("memory_results.json", encoding="utf-8") as f:
            mem_loaded = json.load(f)
        self.assertEqual(time_loaded, time_data)
        self.assertEqual(mem_loaded, mem_data)


class TestTimeMeasurerExtra(unittest.TestCase):
    def test_get_t_value_exact_match(self):
        tm = TimeMeasurer()
        self.assertEqual(tm._get_t_value(6), 2.5706)
        self.assertEqual(tm._get_t_value(11), 2.2281)
        self.assertEqual(tm._get_t_value(101), 1.9840)

    def test_get_t_value_nearest_lower(self):
        tm = TimeMeasurer()
        self.assertEqual(tm._get_t_value(15), 2.2281)

    def test_get_t_value_too_low_raises(self):
        tm = TimeMeasurer()
        with self.assertRaises(ValueError):
            tm._get_t_value(1)


class TestBenchmarkFunctions(unittest.TestCase):
    def test_get_adaptive_n_runs_limits(self):
        self.assertEqual(benchmark.get_adaptive_n_runs(2**10), 101)
        self.assertEqual(benchmark.get_adaptive_n_runs(2**24), 10)

    def test_save_results_creates_files(self):
        time_data = {
            "naive": {
                "best": [
                    {"size": 1024, "time": 0.001, "delta": 0.0001}
                ]
            }
        }
        mem_data = {
            "naive": {
                "best": [
                    {"size": 1024, "memory": 100, "delta": 1}
                ]
            }
        }
        benchmark.save_results(time_data, mem_data)

        self.assertTrue(os.path.exists("time_results.json"))
        self.assertTrue(os.path.exists("memory_results.json"))

        with open("time_results.json", encoding="utf-8") as f:
            loaded_time = json.load(f)
        with open("memory_results.json", encoding="utf-8") as f:
            loaded_mem = json.load(f)

        self.assertEqual(loaded_time, time_data)
        self.assertEqual(loaded_mem, mem_data)


class TestBenchmarkMiniRun(unittest.TestCase):
    def test_single_naive_case_execution(self):
        generator = TestDataGenerator()
        time_measurer = TimeMeasurer()
        memory_measurer = MemoryMeasurer()

        case = generator.generate_all_cases()['naive']['best'][0]
        text, pattern = case
        size = len(text)
        n_runs = benchmark.get_adaptive_n_runs(size)

        # Прогрев
        for _ in range(2):
            naive_search(text, pattern)

        # Измерения
        time, delta = time_measurer.measure(
            naive_search,
            (text, pattern),
            n_runs
        )
        memory, mem_delta = memory_measurer.measure(
            naive_search,
            (text, pattern),
            n_runs
        )

        self.assertGreaterEqual(time, 0)
        self.assertGreaterEqual(memory, 0)


class TestBenchmarkSingleRun(unittest.TestCase):
    def test_benchmark_partial_case_executes(self):
        generator = TestDataGenerator()
        time_measurer = TimeMeasurer()
        memory_measurer = MemoryMeasurer()

        case = generator.generate_all_cases()["naive"]["best"][0]
        text, pattern = case
        size = len(text)
        n_runs = benchmark.get_adaptive_n_runs(size)

        for _ in range(3):
            naive_search(text, pattern)

        time, delta = time_measurer.measure(
            naive_search,
            (text, pattern),
            n_runs
        )
        memory, mem_delta = memory_measurer.measure(
            naive_search,
            (text, pattern),
            n_runs
        )

        self.assertGreaterEqual(time, 0)
        self.assertGreaterEqual(delta, 0)
        self.assertGreaterEqual(memory, 0)


class TestBenchmarkSaveResults(unittest.TestCase):
    def test_save_and_load_results(self):
        time_data = {
            "naive": {
                "best": [{"size": 1, "time": 0.001, "delta": 0.0001}]
            }
        }
        mem_data = {
            "naive": {
                "best": [{"size": 1, "memory": 100, "delta": 1}]
            }
        }
        benchmark.save_results(time_data, mem_data)

        self.assertTrue(os.path.exists("time_results.json"))
        self.assertTrue(os.path.exists("memory_results.json"))

        # Проверка содержимого
        with open("time_results.json", encoding="utf-8") as f:
            loaded_time = json.load(f)
        with open("memory_results.json", encoding="utf-8") as f:
            loaded_mem = json.load(f)

        self.assertEqual(loaded_time, time_data)
        self.assertEqual(loaded_mem, mem_data)

        os.remove("time_results.json")
        os.remove("memory_results.json")


class TestAdaptiveRuns(unittest.TestCase):
    def test_adaptive_n_runs_extremes(self):
        self.assertEqual(benchmark.get_adaptive_n_runs(2**10), 101)
        self.assertEqual(benchmark.get_adaptive_n_runs(2**24), 10)
        self.assertTrue(10 <= benchmark.get_adaptive_n_runs(2**15) <= 101)


class TestBenchmarkUtils(unittest.TestCase):
    def test_get_adaptive_n_runs_limits(self):
        self.assertEqual(benchmark.get_adaptive_n_runs(2**10), 101)
        self.assertEqual(benchmark.get_adaptive_n_runs(2**24), 10)

    def test_save_results_creates_files(self):
        time_data = {
            "test": {
                "best": [{"size": 1, "time": 0.01, "delta": 0.001}]
            }
        }
        mem_data = {
            "test": {
                "best": [{"size": 1, "memory": 123, "delta": 0.5}]
            }
        }
        benchmark.save_results(time_data, mem_data)
        self.assertTrue(os.path.exists("time_results.json"))
        self.assertTrue(os.path.exists("memory_results.json"))
        os.remove("time_results.json")
        os.remove("memory_results.json")


class TestBenchmarkHelpers(unittest.TestCase):
    def test_adaptive_n_runs_boundaries(self):
        self.assertEqual(benchmark.get_adaptive_n_runs(2 ** 10), 101)
        self.assertEqual(benchmark.get_adaptive_n_runs(2 ** 24), 10)

    def test_save_results_creates_and_loads_files(self):
        time_data = {
            "naive": {
                "best": [
                    {"size": 100, "time": 0.01, "delta": 0.001}
                ]
            }
        }
        mem_data = {
            "naive": {
                "best": [
                    {"size": 100, "memory": 123, "delta": 1}
                ]
            }
        }
        benchmark.save_results(time_data, mem_data)

        with open("time_results.json", encoding="utf-8") as f:
            loaded_time = json.load(f)
        with open("memory_results.json", encoding="utf-8") as f:
            loaded_mem = json.load(f)

        self.assertEqual(loaded_time, time_data)
        self.assertEqual(loaded_mem, mem_data)

        os.remove("time_results.json")
        os.remove("memory_results.json")

    def test_save_results_handles_exception(self):
        with patch("builtins.open", side_effect=OSError("fail")):
            try:
                benchmark.save_results({}, {})
            except Exception:
                self.fail("save_results should not raise on open error")


class TestBenchmarkMinimal(unittest.TestCase):
    def test_get_adaptive_n_runs_middle(self):
        self.assertIsInstance(benchmark.get_adaptive_n_runs(2**16), int)

    def test_save_results_handles_io_error(self):
        with patch("builtins.open", side_effect=OSError("fail")):
            try:
                benchmark.save_results({}, {})
            except Exception:
                self.fail("save_results must not raise even if open fails")

    def test_main_reads_missing_files_gracefully(self):
        if os.path.exists("time_results.json"):
            os.remove("time_results.json")
        if os.path.exists("memory_results.json"):
            os.remove("memory_results.json")

        with patch.object(benchmark, "TestDataGenerator") as mock_gen, \
             patch.object(benchmark, "TimeMeasurer") as mock_time, \
             patch.object(benchmark, "MemoryMeasurer") as mock_mem, \
             patch.object(benchmark, "get_adaptive_n_runs", return_value=6):

            instance_gen = mock_gen.return_value
            instance_time = mock_time.return_value
            instance_mem = mock_mem.return_value

            dummy_data = {
                algo: {
                    case: [("A" * 100, "A" * 3)]
                    for case in ["best", "worst", "random"]
                }
                for algo in ["naive", "kmp", "boyer_moore", "rabin_karp",
                             "apostolico_crochemore", "aho_corasick"]
            }

            instance_gen.generate_all_cases.return_value = dummy_data
            instance_time.measure.return_value = (0.001, 0.0001)
            instance_mem.measure.return_value = (100, 1)

            benchmark.main()

        self.assertTrue(os.path.exists("time_results.json"))
        self.assertTrue(os.path.exists("memory_results.json"))

        os.remove("time_results.json")
        os.remove("memory_results.json")


class TestTimeMeasurerEdgeCases(unittest.TestCase):
    def test_measure_raises_on_too_few_runs(self):
        tm = TimeMeasurer()
        text = "A" * 100
        pattern = "A"
        with self.assertRaises(ValueError):
            tm.measure(naive_search, (text, pattern), n_runs=5)


class TestAdvancedSubstringSearchAlgorithms(unittest.TestCase):
    def setUp(self):
        self.algorithms = [
            aho_corasick_search,
            apostolico_crochemore_search
        ]

    def test_match_found(self):
        text = "abracadabra"
        pattern = "cada"
        for algo in self.algorithms:
            with self.subTest(algo=algo.__name__):
                self.assertEqual(algo(text, pattern), 4)

    def test_match_not_found(self):
        text = "abracadabra"
        pattern = "xyz"
        for algo in self.algorithms:
            with self.subTest(algo=algo.__name__):
                self.assertEqual(algo(text, pattern), -1)

    def test_pattern_at_start(self):
        text = "hello world"
        pattern = "hello"
        for algo in self.algorithms:
            with self.subTest(algo=algo.__name__):
                self.assertEqual(algo(text, pattern), 0)

    def test_pattern_at_end(self):
        text = "say goodbye"
        pattern = "goodbye"
        for algo in self.algorithms:
            with self.subTest(algo=algo.__name__):
                self.assertEqual(algo(text, pattern), 4)

    def test_empty_pattern(self):
        text = "something"
        pattern = ""
        for algo in self.algorithms:
            with self.subTest(algo=algo.__name__):
                self.assertEqual(algo(text, pattern), -1)

    def test_empty_text(self):
        text = ""
        pattern = "something"
        for algo in self.algorithms:
            with self.subTest(algo=algo.__name__):
                self.assertEqual(algo(text, pattern), -1)

    def test_both_empty(self):
        text = ""
        pattern = ""
        for algo in self.algorithms:
            with self.subTest(algo=algo.__name__):
                self.assertEqual(algo(text, pattern), -1)


if __name__ == '__main__':
    unittest.main()
