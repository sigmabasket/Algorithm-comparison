import argparse
import json
import logging
import math
from src.algorithms import *
from src.data_generator import TestDataGenerator
from time_measurer import TimeMeasurer
from memory_measurer import MemoryMeasurer
from tqdm import tqdm

logging.basicConfig(
    filename="../results/benchmark.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Проводит измерения времени и памяти "
                    "для алгоритмов поиска подстроки."
    )

    parser.add_argument(
        "-a", "--algorithm",
        type=str,
        choices=["all", "naive", "kmp", "boyer_moore", "rabin_karp",
                 "aho_corasick", "apostolico_croche"],
        default="all",
        help="Алгоритм для тестирования (по умолчанию: all)"
    )

    parser.add_argument(
        "-c", "--case",
        type=str,
        choices=["all", "best", "worst", "random"],
        default="all",
        help="Тип случая данных (по умолчанию: all)"
    )

    return parser.parse_args()


def get_adaptive_n_runs(size: int, min_runs=10, max_runs=101) -> int:
    base = math.log2(size)
    scale = max_runs - min_runs
    factor = (24 - base) / (24 - 10)
    runs = int(min_runs + factor * scale)
    return max(min_runs, min(runs, max_runs))


def save_results(time_results, memory_results):
    try:
        print("Попытка сохранить: time_results.json")
        with open("results/time_results.json", "w", encoding="utf-8") as f:
            json.dump(time_results, f, indent=2, ensure_ascii=False)
        print("Файл сохранён: time_results.json")

        print("Попытка сохранить: memory_results.json")
        with open("results/memory_results.json", "w", encoding="utf-8") as f:
            json.dump(memory_results, f, indent=2, ensure_ascii=False)
        print("Файл сохранён: memory_results.json")

    except Exception as e:
        logging.error(f"Ошибка при сохранении: {e}")


def main():
    args = parse_args()
    selected_algorithms = [args.algorithm] if args.algorithm != "all" else [
        "naive", "kmp", "boyer_moore", "rabin_karp",
        "aho_corasick", "apostolico_croche"
    ]
    if args.case != "all":
        selected_cases = [args.case]
    else:
        selected_cases = ["best", "worst", "random"]
    generator = TestDataGenerator()
    time_measurer = TimeMeasurer()
    memory_measurer = MemoryMeasurer()

    try:
        with open("results/time_results.json", "r", encoding="utf-8") as f:
            time_results = json.load(f)
    except FileNotFoundError:
        time_results = {}

    try:
        with open("results/memory_results.json", "r", encoding="utf-8") as f:
            memory_results = json.load(f)
    except FileNotFoundError:
        memory_results = {}

    algorithms = {
        "naive": naive_search,
        "kmp": kmp_search,
        "boyer_moore": boyer_moore_search,
        "rabin_karp": rabin_karp_search,
        "apostolico_crochemore": apostolico_crochemore_search,
        "aho_corasick": aho_corasick_search
    }

    cases = ["best", "worst", "random"]

    for algo_name, algo_func in algorithms.items():
        if algo_name not in selected_algorithms:
            continue
        print(f"\n--- Тестируем алгоритм: {algo_name} ---")
        if algo_name not in time_results:
            time_results[algo_name] = {}
        if algo_name not in memory_results:
            memory_results[algo_name] = {}

        for case in selected_cases:
            print(f"\n> Случай: {case}")
            if case not in time_results[algo_name]:
                time_results[algo_name][case] = []
            if case not in memory_results[algo_name]:
                memory_results[algo_name][case] = []

            data = generator.generate_all_cases()[algo_name][case]
            print(f"Данных для обработки: {len(data)}")

            for text, pattern in tqdm(data, desc=f"{algo_name} ({case})"):
                size = len(text)
                n_runs = get_adaptive_n_runs(size)
                print(f"  -> size = {size}, n_runs = {n_runs}")

                try:
                    for _ in range(5):  # Прогрев
                        algo_func(text, pattern)

                    time_mean, time_delta = time_measurer.measure(
                        algo_func, (text, pattern), n_runs
                    )
                    memory_mean, memory_delta = memory_measurer.measure(
                        algo_func, (text, pattern), n_runs
                    )

                    time_results[algo_name][case].append({
                        "size": size,
                        "time": time_mean,
                        "delta": time_delta
                    })

                    memory_results[algo_name][case].append({
                        "size": size,
                        "memory": memory_mean,
                        "delta": memory_delta
                    })

                except Exception as e:
                    logging.error(
                        f"Ошибка: {algo_name} ({case}), размер {size}: {e}"
                    )
                    continue

            # Сохраняем после каждого случая
            save_results(time_results, memory_results)


if __name__ == "__main__":
    main()
