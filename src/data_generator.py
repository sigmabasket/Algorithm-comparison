import random
import string
from typing import List, Tuple, Dict


class TestDataGenerator:
    def __init__(self):
        self.sizes = [2**i for i in range(10, 25)]  # 1 КБ — 16 МБ
        self.characters = string.ascii_letters + string.digits

    def generate_all_cases(self) -> Dict[
        str, Dict[str, List[Tuple[str, str]]]
    ]:
        data = {}
        for algo in ['naive', 'kmp', 'boyer_moore', 'rabin_karp',
                     'apostolico_crochemore', 'aho_corasick']:
            data[algo] = {
                'best': [
                    self._generate_best_case(size)
                    for size in self.sizes
                ],
                'worst': [
                    self._generate_worst_case(algo, size)
                    for size in self.sizes
                ],
                'random': [
                    self._generate_random_case(size)
                    for size in self.sizes
                ]
            }
        return data

    def _generate_best_case(self, size: int) -> Tuple[str, str]:
        pattern_length = max(100, size // 10)
        pattern = 'A' * pattern_length
        text = pattern.ljust(size, 'B')
        return text, pattern

    def _generate_worst_case(self, algo: str, size: int) -> Tuple[str, str]:
        if algo == 'naive':
            pattern = 'A' * 999 + 'B'
            text = 'A' * size
        elif algo == 'kmp':
            pattern = 'A' * 1000 + 'B'
            text = self._generate_repeating_pattern(
                'A' * 999 + 'C', size
            )
        elif algo == 'boyer_moore':
            pattern = 'A' * 100 + 'B'
            text = self._generate_repeating_pattern(
                'A' * 100 + 'C', size
            )
        elif algo == 'rabin_karp':
            pattern = "ABC"
            fake_match = "ABC"
            noise = "X"
            base_unit = fake_match[:-1] + noise  # "ABX"
            text = (
                base_unit * (size // len(base_unit) + 1)
            )[:size]
        elif algo == 'apostolico_crochemore':
            pattern = 'A' * 1000 + 'B'
            text = 'A' * size
        elif algo == 'aho_corasick':
            pattern = 'ABC'
            base_unit = 'ABA'
            text = (base_unit * (size // len(base_unit) + 1))[:size]
        return text[:size], pattern

    def _generate_random_case(self, size: int) -> Tuple[str, str]:
        text = ''.join(random.choices(['A', 'B', 'C'], k=size))
        pattern_length = max(10, size // 100)
        if random.random() < 0.3:
            pattern = 'A' * (pattern_length - 1) + 'X'
        else:
            pos = random.randint(0, size - pattern_length)
            pattern = text[pos:pos+pattern_length]
        return text, pattern

    def _generate_repeating_pattern(
        self, pattern: str, target_length: int
    ) -> str:
        return (
            pattern * (target_length // len(pattern) + 1)
        )[:target_length]


# Пример использования
if __name__ == "__main__":
    generator = TestDataGenerator()
    test_data = generator.generate_all_cases()

    # Лучший случай наивного алгоритма для размера 2^10
    print(test_data['naive']['best'][0])
