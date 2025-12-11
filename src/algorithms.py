"""
Модуль algorithms.py: Реализация алгоритмов поиска подстроки в строке.
"""


# Наивный алгоритм
def naive_search(text: str, pattern: str) -> int:
    n, m = len(text), len(pattern)

    if m == 0:
        return 0
    if n < m:
        return -1

    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            return i
    return -1


# Алгоритм Кнута-Морриса-Пратта (KMP)
def kmp_search(text: str, pattern: str) -> int:
    n, m = len(text), len(pattern)

    if m == 0:
        return 0
    if n < m:
        return -1

    def compute_lps(p: str) -> list:
        lps = [0] * len(p)
        length = 0
        for i in range(1, len(p)):
            while length > 0 and p[i] != p[length]:
                length = lps[length - 1]
            if p[i] == p[length]:
                length += 1
                lps[i] = length
        return lps

    lps = compute_lps(pattern)
    i = j = 0
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == m:
                return i - j
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1


# Алгоритм Бойера-Мура
def boyer_moore_search(text: str, pattern: str) -> int:
    n, m = len(text), len(pattern)

    if m == 0:
        return 0
    if n < m:
        return -1

    def bad_char_heuristic(p: str) -> dict:
        return {p[i]: i for i in range(len(p))}

    bad_char = bad_char_heuristic(pattern)
    s = 0
    while s <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1
        if j < 0:
            return s
        else:
            s += max(1, j - bad_char.get(text[s + j], -1))
    return -1


# Алгоритм Рабина-Карпа
def rabin_karp_search(
    text: str,
    pattern: str,
    d: int = 256,
    q: int = 101
) -> int:
    n, m = len(text), len(pattern)

    if m == 0:
        return 0
    if n < m:
        return -1

    h_pattern = h_window = 0
    h = pow(d, m - 1, q)

    for i in range(m):
        h_pattern = (d * h_pattern + ord(pattern[i])) % q
        h_window = (d * h_window + ord(text[i])) % q

    for i in range(n - m + 1):
        if h_pattern == h_window:
            if text[i:i + m] == pattern:
                return i
        if i < n - m:
            h_window = (
                d * (h_window - ord(text[i]) * h)
                + ord(text[i + m])
            ) % q
            if h_window < 0:
                h_window += q
    return -1


def apostolico_crochemore_search(text: str, pattern: str) -> int:
    n, m = len(text), len(pattern)
    if m == 0 or n == 0:
        return -1
    if m > n:
        return -1

    # Построение префиксной функции (LPS)
    def compute_lps(p):
        lps = [0] * m
        length = 0
        for i in range(1, m):
            while length > 0 and p[i] != p[length]:
                length = lps[length - 1]
            if p[i] == p[length]:
                length += 1
                lps[i] = length
        return lps

    lps = compute_lps(pattern)
    shift = 0
    j = 0
    while shift <= n - m:
        while j < m and pattern[j] == text[shift + j]:
            j += 1
        if j == m:
            return shift
        if j == 0:
            shift += 1
        else:
            shift += max(1, j - lps[j - 1])
        j = 0
    return -1


def aho_corasick_search(text: str, pattern: str) -> int:
    from collections import deque

    class Node:
        def __init__(self):
            self.children = {}
            self.fail = None
            self.output = []

    def build_trie(pattern):
        root = Node()
        node = root
        for char in pattern:
            if char not in node.children:
                node.children[char] = Node()
            node = node.children[char]
        node.output.append(0)
        return root

    def build_failure_links(root):
        queue = deque()
        for child in root.children.values():
            child.fail = root
            queue.append(child)

        while queue:
            current = queue.popleft()
            for key, child in current.children.items():
                fail = current.fail
                while fail and key not in fail.children:
                    fail = fail.fail
                if fail and key in fail.children:
                    child.fail = fail.children[key]
                else:
                    child.fail = root
                child.output += child.fail.output
                queue.append(child)

    root = build_trie(pattern)
    build_failure_links(root)

    node = root
    for i, c in enumerate(text):
        while node and c not in node.children:
            node = node.fail
        if not node:
            node = root
            continue
        node = node.children[c]
        if node.output:
            return i - len(pattern) + 1
    return -1


if __name__ == "__main__":
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"

    print("Тестирование алгоритмов:")
    print(f"Текст: '{text}'\nПаттерн: '{pattern}'\n")

    # Тестируем все алгоритмы
    algorithms = [
        ("Наивный алгоритм", naive_search),
        ("KMP", kmp_search),
        ("Boyer-Moore", boyer_moore_search),
        ("Rabin-Karp", rabin_karp_search)
    ]

    for name, algo in algorithms:
        result = algo(text, pattern)
        status = 'Найден' if result != -1 else 'Не найден'
        print(f"{name}: {status} (индекс {result})")
