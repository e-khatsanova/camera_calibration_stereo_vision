import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


CAMERA = "iphone"
TARGET = "a4"
PATTERN = "chess"

RESULTS_DIR = Path("calib_results")


def main():

    files = RESULTS_DIR.glob(
        f"{CAMERA}_{PATTERN}_{TARGET}_n*.npz"
    )

    results = []

    # Читаем все результаты
    for file in files:

        data = np.load(file)

        n = int(data["n_images"])

        min_rms = float(data["min_rms"])
        median_rms = float(data["median_rms"])
        max_rms = float(data["max_rms"])

        results.append(
            (n, min_rms, median_rms, max_rms)
        )

    if not results:
        raise RuntimeError(
            "Не найдены файлы с результатами."
        )

    # Сортируем по количеству кадров
    results.sort(
        key=lambda x: x[0]
    )

    n = [item[0] for item in results]
    min_rms = [item[1] for item in results]
    median_rms = [item[2] for item in results]
    max_rms = [item[3] for item in results]

    # График
 

    plt.figure(figsize=(9, 5))

    # Медианное значение
    plt.plot(
        n,
        median_rms,
        marker="o",
        label="Median RMS"
    )

    # Минимум
    plt.plot(
        n,
        min_rms,
        linestyle="--",
        label="Min RMS"
    )

    # Максимум
    plt.plot(
        n,
        max_rms,
        linestyle="--",
        label="Max RMS"
    )

    plt.xlabel("Количество кадров")
    plt.ylabel("RMS reprojection error, px")

    plt.title(
        f"{CAMERA} — {PATTERN} — {TARGET}"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()


    # Сохраняем


    output_file = (
        RESULTS_DIR
        / f"{CAMERA}_{PATTERN}_{TARGET}_rms.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    print(
        f"График сохранён: {output_file}"
    )

    plt.show()


if __name__ == "__main__":
    main()