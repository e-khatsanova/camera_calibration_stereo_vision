"""
Исследование влияния количества изображений на калибровку камеры
с использованием Asymmetric Circles Grid.

Для каждого количества кадров:

    N = 10, 11, 12, ..., 30

выполняются до 31 различных калибровок на случайных комбинациях
из N изображений.

Для каждой комбинации получаем:

    - RMS reprojection error;
    - camera_matrix;
    - distortion coefficients.

Результаты сортируются по RMS.

Сохраняются три характерных результата:

    MIN    - минимальная RMS;
    MEDIAN - медианная RMS;
    MAX    - максимальная RMS.

Основная цель — исследовать стабильность внутренних параметров камеры
при увеличении количества используемых изображений.
"""


import cv2
import numpy as np

from pathlib import Path
from itertools import combinations
from math import comb



CAMERA = "iphone"
TARGET = "a4"
PATTERN = "acircle"

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



# Количество ЦЕНТРОВ кругов:
# (columns, rows) смотрим в файле генерации

PATTERN_SIZE = (4, 9)

# Расстояние между соседними позициями сетки.
#
# В asymmetric circles grid соседние строки смещены
# относительно друг друга на половину горизонтального шага
# то есть мы смотрим на расстояние между центрами двух кругов в строке
# и делим его пополам (как положено друзьям).
# Здесь обязательно указать реальное значение мишени.
# Фотки с ними есть в data\mesurments
# tab = 13.5
# a4 = 18.0

POINT_DISTANCE = 18.0  # мм


# Для обычной symmetric grid координаты выглядели бы:
#
# row 0:  0   1   2   3
# row 1:  0   1   2   3
# row 2:  0   1   2   3
#
# У asymmetric grid строки смещены:
#
# row 0:  0       2       4       6
# row 1:      1       3       5       7
# row 2:  0       2       4       6
# row 3:      1       3       5       7
#
# Именно такое расположение ожидает OpenCV.

object_points_template = np.zeros(
    (
        PATTERN_SIZE[0] * PATTERN_SIZE[1],
        3
    ),
    dtype=np.float32
)

point_index = 0

for row in range(PATTERN_SIZE[1]):

    for column in range(PATTERN_SIZE[0]):

        object_points_template[
            point_index,
            0
        ] = (
            2 * column
            + row % 2
        ) * POINT_DISTANCE

        object_points_template[
            point_index,
            1
        ] = (
            row
            * POINT_DISTANCE
        )

        point_index += 1



# КАЛИБРОВКА ОДНОЙ КОМБИНАЦИИ
def calibrate_frames(
    selected_frames,
    image_size
):

    object_points_all = []
    image_points_all = []

    used_images = []

    for item in selected_frames:

        object_points_all.append(
            object_points_template.copy()
        )

        image_points_all.append(
            item["centers"].astype(
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


# СОЗДАНИЕ КОМБИНАЦИЙ
def make_combinations(
    number_of_frames,
    n_images,
    max_combinations,
    rng
):

    total_possible = comb(
        number_of_frames,
        n_images
    )

    number_to_use = min(
        max_combinations,
        total_possible
    )

    # Если возможных комбинаций меньше 31,
    # используем их все.

    if total_possible <= max_combinations:

        return list(
            combinations(
                range(number_of_frames),
                n_images
            )
        )

    # Иначе случайно выбираем
    # 31 уникальную комбинацию.

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
        f"Всего изображений: "
        f"{len(image_paths)}"
    )


    # ИЩЕМ ASYMMETRIC CIRCLES GRID
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

        if image_size is None:

            image_size = current_size

        elif current_size != image_size:

            print(
                f"Пропущено из-за другого размера: "
                f"{image_path.name}"
            )

            continue


        # ПОИСК ЦЕНТРОВ КРУГОВ
        found, centers = cv2.findCirclesGrid(
            gray,
            PATTERN_SIZE,
            flags=cv2.CALIB_CB_ASYMMETRIC_GRID
        )

        if not found:

            print(
                f"Asymmetric Circles Grid не найден: "
                f"{image_path.name}"
            )

            continue

        valid_frames.append(
            {
                "path": image_path,
                "centers": centers,
            }
        )

        print(
            f"OK: "
            f"{image_path.name} — "
            f"{len(centers)} точек"
        )

    print()

    print(
        f"Asymmetric Circles Grid найден на "
        f"{len(valid_frames)} из "
        f"{len(image_paths)} изображений."
    )

    if len(valid_frames) < N_MIN:

        raise RuntimeError(
            f"Найдено только "
            f"{len(valid_frames)} валидных кадров. "
            f"Нужно минимум {N_MIN}."
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

        # КАЛИБРУЕМ ВСЕ КОМБИНАЦИИ
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


        # СОХРАНЯЕМ
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

            max_rms=result_max[
                "rms"
            ],

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

            # Параметры asymmetric circles

            pattern_size=np.array(
                PATTERN_SIZE
            ),

            point_distance=POINT_DISTANCE,
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