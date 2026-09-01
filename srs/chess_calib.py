"""
Исследование влияния количества изображений на калибровку камеры Chessboard.

ЗАДАЧА
------
Мы хотим определить, насколько устойчиво оцениваются внутренние параметры камеры
при различном количестве изображений калибровочной мишени Chessboard.

Для каждого количества кадров:

    N = 10, 11, 12, ..., 30

мы НЕ выполняем калибровку только на одной конкретной группе изображений.

Вместо этого:

1. Загружаем все изображения из папки.

2. На каждом изображении ищем внутренние углы Chessboard.

3. Изображения, на которых Chessboard не распознан, отбрасываем.

4. Из оставшихся валидных изображений для каждого N
   формируем до 31 РАЗЛИЧНОЙ случайной комбинации по N кадров.

5. Для каждой комбинации независимо выполняем:

       cv2.calibrateCamera()

   и получаем:

       - RMS reprojection error;
       - camera_matrix;
       - distortion coefficients.

6. Все результаты сортируем по RMS reprojection error.

7. Из них сохраняем три характерных результата:

       MIN
       комбинация с минимальной RMS ошибкой;

       MEDIAN
       комбинация, находящаяся в середине
       отсортированного списка;

       MAX
       комбинация с максимальной RMS ошибкой.

Почему именно 31 комбинация?
----------------------------
31 — нечётное число, поэтому после сортировки существует один
однозначный медианный результат: 16-й результат из 31.

Что мы хотим увидеть?
---------------------
Если количества кадров недостаточно, разные комбинации одних и тех же N кадров
могут давать заметно разные значения внутренних параметров:

    fx
    fy
    cx
    cy

При увеличении N ожидается, что результаты разных комбинаций
будут становиться ближе друг к другу.

Нас интересует не только RMS ошибка,
но прежде всего СТАБИЛЬНОСТЬ внутренних параметров камеры.

ВАЖНО
------
Количество уникальных комбинаций ограничено:

    C(M, N)

где M — общее количество валидных изображений.

Например, если валидных изображений ровно 30:

    N = 30 -> существует только 1 комбинация;
    N = 29 -> существует только 30 комбинаций.

В таких случаях программа использует все существующие комбинации.

Для воспроизводимости используется фиксированный RANDOM_SEED.
"""


import cv2
import numpy as np

from pathlib import Path
from itertools import combinations
from math import comb



CAMERA = "iphone"
TARGET = "a4"
PATTERN = "chess"

N_MIN = 10
N_MAX = 30

N_COMBINATIONS = 31

RANDOM_SEED = 42


IMAGES_DIR = (
    Path("data")
    / CAMERA
    / f"{PATTERN}_{TARGET}"
)

OUTPUT_DIR = Path("calib_results")



# ПАРАМЕТРЫ CHESSBOARD


# ВАЖНО:
# здесь указывается количество ВНУТРЕННИХ углов,
# а не количество клеток.
#
# Например:
#
#     9 × 6 клеток
#
# соответствует:
#
#     8 × 5 внутренних углов

PATTERN_SIZE = (8, 5)

# Физический размер стороны одной клетки.
# Здесь обязательно указать реальное значение мишени.
# Фотки с ними есть в data\mesurments
# tab = 13.5
# a4 = 18.0
SQUARE_SIZE = 18.0  # мм


# Все точки лежат в одной плоскости:
#
# Z = 0
#
# Получаем:
#
# (0,0,0)
# (1,0,0)
# (2,0,0)
# ...
#
# затем умножаем на физический размер клетки.

object_points_template = np.zeros(
    (
        PATTERN_SIZE[0] * PATTERN_SIZE[1],
        3
    ),
    dtype=np.float32
)

object_points_template[:, :2] = (
    np.mgrid[
        0:PATTERN_SIZE[0],
        0:PATTERN_SIZE[1]
    ]
    .T
    .reshape(-1, 2)
)

object_points_template *= SQUARE_SIZE


def calibrate_frames(
    selected_frames,
    image_size
):
    """
    Выполняет одну калибровку камеры
    на конкретной комбинации кадров.
    """

    object_points_all = []
    image_points_all = []

    used_images = []

    for item in selected_frames:

        object_points_all.append(
            object_points_template.copy()
        )

        image_points_all.append(
            item["corners"].astype(
                np.float32
            )
        )

        used_images.append(
            item["path"].name
        )

    (
        rms_error,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs
    ) = cv2.calibrateCamera(
        objectPoints=object_points_all,
        imagePoints=image_points_all,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    return {
        "rms": rms_error,

        "camera_matrix": camera_matrix,

        "dist_coeffs": dist_coeffs,

        "rvecs": rvecs,

        "tvecs": tvecs,

        "used_images": used_images,
    }



def make_combinations(
    number_of_frames,
    n_images,
    max_combinations,
    rng
):
    """
    Создаёт различные комбинации индексов кадров.

    Если возможных комбинаций меньше 31,
    используются все.

    Если комбинаций больше,
    случайно выбираются 31 уникальная комбинация.
    """

    total_possible = comb(
        number_of_frames,
        n_images
    )

    number_to_use = min(
        max_combinations,
        total_possible
    )

    # Если комбинаций мало —
    # просто используем их все.

    if total_possible <= max_combinations:

        return list(
            combinations(
                range(number_of_frames),
                n_images
            )
        )

    # Если вариантов много —
    # случайно выбираем уникальные комбинации.

    selected_combinations = set()

    while len(selected_combinations) < number_to_use:

        indices = rng.choice(
            number_of_frames,
            size=n_images,
            replace=False
        )

        indices = tuple(
            sorted(indices)
        )

        selected_combinations.add(
            indices
        )

    return list(
        selected_combinations
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    image_paths = sorted([
        *IMAGES_DIR.glob("*.jpg"),
        *IMAGES_DIR.glob("*.jpeg"),
        *IMAGES_DIR.glob("*.png"),
    ])

    if not image_paths:

        raise RuntimeError(
            f"В папке нет изображений: "
            f"{IMAGES_DIR}"
        )

    print()

    print(
        f"Всего изображений в папке: "
        f"{len(image_paths)}"
    )


    # ИЩЕМ CHESSBOARD НА ВСЕХ ИЗОБРАЖЕНИЯХ

    valid_frames = []

    image_size = None

    for image_path in image_paths:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"Не удалось открыть: "
                f"{image_path.name}"
            )

            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        current_size = (
            gray.shape[1],
            gray.shape[0]
        )

        # Проверяем разрешение изображений.

        if image_size is None:

            image_size = current_size

        elif current_size != image_size:

            print(
                f"Пропущено из-за другого размера: "
                f"{image_path.name}"
            )

            continue


        # ИЩЕМ ВНУТРЕННИЕ УГЛЫ CHESSBOARD
        found, corners = (
            cv2.findChessboardCornersSB(
                gray,
                PATTERN_SIZE
            )
        )

        if not found:

            print(
                f"Chessboard не найден: "
                f"{image_path.name}"
            )

            continue

        # Chessboard успешно найден.

        valid_frames.append(
            {
                "path": image_path,
                "corners": corners,
            }
        )

        print(
            f"OK: "
            f"{image_path.name} — "
            f"{len(corners)} углов"
        )


    # ПРОВЕРЯЕМ КОЛИЧЕСТВО ВАЛИДНЫХ ИЗОБРАЖЕНИЙ
    
    print(
        f"Chessboard успешно распознан на "
        f"{len(valid_frames)} из "
        f"{len(image_paths)} изображений."
    )

    if len(valid_frames) < N_MIN:

        raise RuntimeError(
            f"Для эксперимента требуется минимум "
            f"{N_MIN} валидных изображений. "
            f"Найдено только "
            f"{len(valid_frames)}."
        )

    actual_n_max = min(
        N_MAX,
        len(valid_frames)
    )


    # ЭКСПЕРИМЕНТ N = 10 ... 30
    for n_images in range(
        N_MIN,
        actual_n_max + 1
    ):

        print()
        print("=" * 70)

        print(
            f"N = {n_images}"
        )

        print("=" * 70)


        # СОЗДАЁМ КОМБИНАЦИИ
        frame_combinations = make_combinations(
            number_of_frames=len(
                valid_frames
            ),

            n_images=n_images,

            max_combinations=N_COMBINATIONS,

            rng=rng
        )

        print(
            f"Будет выполнено калибровок: "
            f"{len(frame_combinations)}"
        )

        calibration_results = []

        # КАЛИБРУЕМ КАЖДУЮ КОМБИНАЦИЮ
        for combination_number, indices in enumerate(
            frame_combinations,
            start=1
        ):

            selected_frames = [
                valid_frames[i]
                for i in indices
            ]

            result = calibrate_frames(
                selected_frames,
                image_size
            )

            calibration_results.append(
                result
            )

            print(
                f"{combination_number:02d}/"
                f"{len(frame_combinations):02d}  "
                f"RMS = "
                f"{result['rms']:.6f} px"
            )

        # СОРТИРУЕМ ПО RMS
        calibration_results.sort(
            key=lambda result:
            result["rms"]
        )

        result_min = (
            calibration_results[0]
        )

        result_max = (
            calibration_results[-1]
        )

        median_index = (
            len(calibration_results)
            // 2
        )

        result_median = (
            calibration_results[
                median_index
            ]
        )


        print()

        print(
            f"N = {n_images}"
        )

        print(
            f"MIN RMS    = "
            f"{result_min['rms']:.6f} px"
        )

        print(
            f"MEDIAN RMS = "
            f"{result_median['rms']:.6f} px"
        )

        print(
            f"MAX RMS    = "
            f"{result_max['rms']:.6f} px"
        )

        print()

        print("fx:")

        print(
            f"  min    = "
            f"{result_min['camera_matrix'][0, 0]:.3f}"
        )

        print(
            f"  median = "
            f"{result_median['camera_matrix'][0, 0]:.3f}"
        )

        print(
            f"  max    = "
            f"{result_max['camera_matrix'][0, 0]:.3f}"
        )

        print()

        print("fy:")

        print(
            f"  min    = "
            f"{result_min['camera_matrix'][1, 1]:.3f}"
        )

        print(
            f"  median = "
            f"{result_median['camera_matrix'][1, 1]:.3f}"
        )

        print(
            f"  max    = "
            f"{result_max['camera_matrix'][1, 1]:.3f}"
        )


        # СОХРАНЕНИЕ
        output_file = (
            OUTPUT_DIR
            / f"{CAMERA}_{PATTERN}_{TARGET}_n{n_images}.npz"
        )

        np.savez(
            output_file,

            # Общая информация

            n_images=n_images,

            n_combinations=len(
                calibration_results
            ),

            image_size=np.array(
                image_size
            ),

            # MIN

            min_rms=result_min["rms"],

            min_camera_matrix=result_min[
                "camera_matrix"
            ],

            min_dist_coeffs=result_min[
                "dist_coeffs"
            ],

            min_used_images=np.array(
                result_min[
                    "used_images"
                ]
            ),

            # MEDIAN

            median_rms=result_median[
                "rms"
            ],

            median_camera_matrix=result_median[
                "camera_matrix"
            ],

            median_dist_coeffs=result_median[
                "dist_coeffs"
            ],

            median_used_images=np.array(
                result_median[
                    "used_images"
                ]
            ),

            # MAX

            max_rms=result_max["rms"],

            max_camera_matrix=result_max[
                "camera_matrix"
            ],

            max_dist_coeffs=result_max[
                "dist_coeffs"
            ],

            max_used_images=np.array(
                result_max[
                    "used_images"
                ]
            ),

            # Все RMS

            all_rms=np.array([
                result["rms"]
                for result
                in calibration_results
            ]),

            # Параметры Chessboard

            pattern_size=np.array(
                PATTERN_SIZE
            ),

            square_size=SQUARE_SIZE,
        )

        print()

        print(
            f"Сохранено: "
            f"{output_file}"
        )

    print()
    print("=" * 70)
    print("Эксперимент завершён.")
    print("=" * 70)


if __name__ == "__main__":
    main()