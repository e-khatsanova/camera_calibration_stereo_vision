"""
Исследование влияния количества изображений на калибровку камеры ChArUco.

ЗАДАЧА
------
Мы (я) хотим определить, насколько устойчиво оцениваются внутренние параметры камеры
при различном количестве изображений калибровочной мишени.

Для каждого количества кадров:

    N = 10, 11, 12, ..., 30

мы НЕ выполняем калибровку только на одной конкретной группе изображений 
(мы пробовали и получили плохо интерпретируемые данные).

Вместо этого:

1. Загружаем все изображения из папки.

2. На каждом изображении ищем ChArUco.

3. Изображения, на которых ChArUco не распознан
   или найдено слишком мало углов, отбрасываем.

4. Из оставшихся валидных изображений для каждого N
   формируем до 31 РАЗЛИЧНОЙ случайной комбинации по N кадров.

   Например:

       N = 10

       комбинация 1:
       frame_001, frame_004, frame_007, ...

       комбинация 2:
       frame_000, frame_003, frame_009, ...

       ...

       комбинация 31

5. Для каждой комбинации независимо выполняем:

       cv2.calibrateCamera()

   и получаем:

       - RMS reprojection error;
       - camera_matrix;
       - distortion coefficients.

6. Все 31 результата сортируем по RMS reprojection error.

7. Из них сохраняем три характерных результата:

       MIN
       комбинация с минимальной RMS ошибкой;

       MEDIAN
       комбинация, находящаяся ровно в середине
       отсортированного списка результатов;

       MAX
       комбинация с максимальной RMS ошибкой.

Почему именно 31 комбинация?

31 — нечётное число, поэтому после сортировки существует один однозначный
медианный результат: 16-й результат из 31 (не будем усложнять и так сложную задачу).

Что мы хотим увидеть?

Если количества кадров недостаточно, разные комбинации одних и тех же N кадров
могут давать заметно разные значения:

    fx
    fy
    cx
    cy

При увеличении N ожидается, что результаты разных комбинаций будут становиться
ближе друг к другу.

То есть нас интересует не только уменьшение RMS ошибки,
но прежде всего СТАБИЛЬНОСТЬ внутренних параметров камеры.

ВАЖНО
Количество уникальных комбинаций ограничено:

    C(M, N)

где M — общее количество валидных изображений.

Например, если валидных изображений ровно 30:

    N = 30 -> существует только 1 комбинация;
    N = 29 -> существует только 30 комбинаций.

В таких случаях программа использует все существующие уникальные комбинации,
а не пытается искусственно создать 31 одинаковый набор.

Для воспроизводимости используется фиксированный RANDOM_SEED.
При повторном запуске будут выбраны те же самые случайные комбинации (вдруг вы тоже захотите повторить).
"""


import cv2
import numpy as np

from pathlib import Path
from itertools import combinations
from math import comb



CAMERA = "iphone"
TARGET = "tab"
PATTERN = "charuco"

# Исследуем N от 10 до 30 включительно
N_MIN = 10
N_MAX = 30

# Максимальное количество различных комбинаций для каждого N
N_COMBINATIONS = 31

# Фиксированный seed нужен для воспроизводимости эксперимента
RANDOM_SEED = 42



IMAGES_DIR = (
    Path("data")
    / CAMERA
    / f"{PATTERN}_{TARGET}"
)

OUTPUT_DIR = Path("calib_results")



# ПАРАМЕТРЫ МОЕЙ ChArUco-ДОСКИ (можно посмотреть файл в котором я ее генерировала)

SQUARES_X = 5
SQUARES_Y = 7

SQUARE_SIZE = 30.0      # мм
MARKER_SIZE = 20.0      # мм

ARUCO_DICT = cv2.aruco.DICT_5X5_100

'''
dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_5X5_100
)

board = cv2.aruco.CharucoBoard(
    (5, 7),
    30.0,   # square size
    20.0,   # marker size
    dictionary
)
'''


def calibrate_frames(selected_frames, board, image_size):
    """
    Выполняет одну калибровку камеры.

    selected_frames:
        конкретная комбинация из N изображений.

    Возвращает:
        RMS ошибку,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
        имена использованных изображений.
    """

    object_points_all = []
    image_points_all = []

    used_images = []

    for item in selected_frames:

        object_points, image_points = board.matchImagePoints(
            item["corners"],
            item["ids"]
        )

        object_points_all.append(
            object_points.astype(np.float32)
        )

        image_points_all.append(
            image_points.astype(np.float32)
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

    Например, если:

        number_of_frames = 30
        n_images = 10

    функция возвращает максимум 31 различную комбинацию
    по 10 индексов.

    Если всего возможных комбинаций меньше 31,
    возвращаются ВСЕ возможные комбинации.
    """

    total_possible = comb(
        number_of_frames,
        n_images
    )

    number_to_use = min(
        max_combinations,
        total_possible
    )


    if total_possible <= max_combinations:

        return list(
            combinations(
                range(number_of_frames),
                n_images
            )
        )

    selected_combinations = set()

    while len(selected_combinations) < number_to_use:

        indices = rng.choice(
            number_of_frames,
            size=n_images,
            replace=False
        )

        # Сортируем индексы, чтобы
        # (1, 2, 3) и (3, 2, 1)
        # считались одной комбинацией.
        indices = tuple(
            sorted(indices)
        )

        selected_combinations.add(
            indices
        )

    return list(selected_combinations)




















def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    # Генератор случайных чисел.
    # Seed фиксированный, поэтому эксперимент воспроизводим.

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    dictionary = cv2.aruco.getPredefinedDictionary(
        ARUCO_DICT
    )

    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_SIZE,
        MARKER_SIZE,
        dictionary
    )

    detector = cv2.aruco.CharucoDetector(
        board
    )

    image_paths = sorted([
        *IMAGES_DIR.glob("*.jpg"),
        *IMAGES_DIR.glob("*.jpeg"),
        *IMAGES_DIR.glob("*.png"),
    ])

    if not image_paths:
        raise RuntimeError(
            f"В папке нет изображений: {IMAGES_DIR}"
        )

    print()
    print(
        f"Всего изображений в папке: "
        f"{len(image_paths)}"
    )

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

        (
            charuco_corners,
            charuco_ids,
            _,
            _
        ) = detector.detectBoard(
            gray
        )

        if charuco_ids is None:

            print(
                f"ChArUco не найден: "
                f"{image_path.name}"
            )

            continue

        # Четыре точки минимальный фильтр.


        if len(charuco_ids) < 4:

            print(
                f"Слишком мало углов "
                f"({len(charuco_ids)}): "
                f"{image_path.name}"
            )

            continue

        valid_frames.append(
            {
                "path": image_path,
                "corners": charuco_corners,
                "ids": charuco_ids,
            }
        )

        print(
            f"OK: "
            f"{image_path.name} — "
            f"{len(charuco_ids)} углов"
        )



    print()
    print(
        f"ChArUco успешно распознан на "
        f"{len(valid_frames)} из "
        f"{len(image_paths)} изображений."
    )

    if len(valid_frames) < N_MIN:

        raise RuntimeError(
            f"Для эксперимента требуется минимум "
            f"{N_MIN} валидных изображений. "
            f"Найдено только {len(valid_frames)}."
        )

    # Нельзя исследовать N больше количества валидных кадров.
    actual_n_max = min(
        N_MAX,
        len(valid_frames)
    )



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

        # Создаём различные комбинации по N кадров.

        frame_combinations = make_combinations(
            number_of_frames=len(valid_frames),
            n_images=n_images,
            max_combinations=N_COMBINATIONS,
            rng=rng
        )

        print(
            f"Будет выполнено калибровок: "
            f"{len(frame_combinations)}"
        )

        # Результаты всех комбинаций этого N.


        calibration_results = []


        # Калибруем камеру на каждой комбинации.


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
                board,
                image_size
            )

            calibration_results.append(
                result
            )

            print(
                f"  "
                f"{combination_number:02d}/"
                f"{len(frame_combinations):02d}"
                f"  RMS = "
                f"{result['rms']:.6f} px"
            )


        # СОРТИРУЕМ ПО RMS

        calibration_results.sort(
            key=lambda result: result["rms"]
        )


        # Минимальная ошибка.


        result_min = calibration_results[0]


        # Максимальная ошибка.


        result_max = calibration_results[-1]


        # Медианная ошибка.
        # При 31 результатах:
        # index = 15
        # то есть 16-й элемент.
 

        median_index = (
            len(calibration_results) // 2
        )

        result_median = calibration_results[
            median_index
        ]


        # ВЫВОДИМ РЕЗУЛЬТАТЫ


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

        print(
            "fx:"
        )

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

        print(
            "fy:"
        )

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

  
        # СОХРАНЯЕМ РЕЗУЛЬТАТ ДЛЯ ЭТОГО N

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


            # MIN RMS

            min_rms=result_min["rms"],

            min_camera_matrix=result_min[
                "camera_matrix"
            ],

            min_dist_coeffs=result_min[
                "dist_coeffs"
            ],

            min_used_images=np.array(
                result_min["used_images"]
            ),

            # MEDIAN RMS

            median_rms=result_median["rms"],

            median_camera_matrix=result_median[
                "camera_matrix"
            ],

            median_dist_coeffs=result_median[
                "dist_coeffs"
            ],

            median_used_images=np.array(
                result_median["used_images"]
            ),

            # MAX RMS

            max_rms=result_max["rms"],

            max_camera_matrix=result_max[
                "camera_matrix"
            ],

            max_dist_coeffs=result_max[
                "dist_coeffs"
            ],

            max_used_images=np.array(
                result_max["used_images"]
            ),

            # Дополнительно сохраняем RMS ВСЕХ комбинаций.
            # Это пригодится позже для построения распределения ошибки и оценки разброса.


            all_rms=np.array([
                result["rms"]
                for result
                in calibration_results
            ]),


            # Параметры ChArUco

            squares_x=SQUARES_X,
            squares_y=SQUARES_Y,

            square_size=SQUARE_SIZE,
            marker_size=MARKER_SIZE,
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