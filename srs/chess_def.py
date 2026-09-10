import cv2
import numpy as np

from pathlib import Path


def calibrate_chess(images_dir, pattern_size, square_size):

    images_dir = Path(images_dir)

    object_points_template = np.zeros(
        (
            pattern_size[0] * pattern_size[1],
            3
        ),
        dtype=np.float32
    )

    object_points_template[:, :2] = (
        np.mgrid[
            0:pattern_size[0],
            0:pattern_size[1]
        ]
        .T
        .reshape(-1, 2)
    )

    object_points_template *= square_size

    image_paths = sorted([
        *images_dir.glob("*.jpg"),
        *images_dir.glob("*.jpeg"),
        *images_dir.glob("*.png"),
    ])

    m = len(image_paths)

    if not image_paths:
        raise RuntimeError(
            f"В папке нет изображений: {images_dir}"
        )

    object_points_all = []
    image_points_all = []

    image_size = None

    for image_path in image_paths:

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Не удалось открыть: {image_path.name}")
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
            '''
            print(
                f"Пропущено из-за другого размера: "
                f"{image_path.name}"
            )
            '''
            continue

        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size
        )

        if not found:
            '''
            print(
                f"Chessboard не найден: "
                f"{image_path.name}"
            )
            '''
            continue

        '''
        print(
            f"OK: {image_path.name} — "
            f"{len(corners)} углов"
        )
        '''

        object_points_all.append(
            object_points_template.copy()
        )

        image_points_all.append(
            corners.astype(np.float32)
        )

    n = len(object_points_all)

    if n < 2:
        raise RuntimeError(
            "Недостаточно изображений "
            "с распознанным Chessboard."
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objectPoints=object_points_all,
        imagePoints=image_points_all,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    return camera_matrix, dist_coeffs, rms, n, m