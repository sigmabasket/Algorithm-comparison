import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _power_fit_curve(x: np.ndarray, y: np.ndarray) -> tuple:
    """Аппроксимация степенной функцией
    через логарифмическое преобразование."""
    log_x = np.log(x)
    log_y = np.log(y)

    slope, intercept = np.polyfit(log_x, log_y, 1)
    a = np.exp(intercept)
    b = slope

    y_pred = a * x**b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    label = f"y = {a:.2e}x^{b:.2f} (R²={r2:.2f})"
    return (a, b), label


def plot_time_results_power_only() -> None:
    # Читаем единую таблицу
    df = pd.read_csv("results.csv", encoding="utf-8-sig")

    plt.figure(figsize=(18, 6))
    cases = ["best", "worst", "random"]

    algorithms = [
        "naive",
        "kmp",
        "boyer_moore",
        "rabin_karp",
        "apostolico_crochemore",
        "aho_corasick",
    ]
    colors = {
        "naive": "blue",
        "kmp": "green",
        "boyer_moore": "red",
        "rabin_karp": "purple",
        "apostolico_crochemore": "orange",
        "aho_corasick": "brown",
    }

    for idx, case in enumerate(cases, 1):
        plt.subplot(1, 3, idx)

        for algo in algorithms:
            subset = df[(df["Алгоритм"] == algo) & (df["Случай"] == case)]
            if subset.empty:
                continue
            subset = subset.sort_values("Размер (байт)")

            x = subset["Размер (байт)"].to_numpy()
            y = subset["Время (сек)"].to_numpy()
            yerr = subset["ΔВремя"].to_numpy()

            # экспериментальные точки
            plt.errorbar(
                x, y, yerr=yerr,
                fmt="o",
                capsize=5,
                color=colors[algo],
                label=f"{algo} (эксперимент)",
                alpha=0.7,
            )

            # аппроксимация степенной кривой
            if len(x) > 1:
                (a, b), label = _power_fit_curve(x, y)
                x_fit = np.linspace(x.min(), x.max(), 200)
                y_fit = a * x_fit**b
                plt.plot(
                    x_fit, y_fit,
                    "--",
                    color=colors[algo],
                    label=f"{algo} ({label})",
                )

        plt.title(f"{case.capitalize()} случаи")
        plt.xlabel("Размер данных (байты, log)")
        plt.ylabel("Время (сек, log)")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, which="both", ls=":")
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.figtext(
        0.5, -0.02,
        "Лог‑лог график: степенные зависимости отображаются прямыми линиями",
        wrap=True, horizontalalignment="center", fontsize=9,
    )
    plt.show()


if __name__ == "__main__":
    plot_time_results_power_only()
