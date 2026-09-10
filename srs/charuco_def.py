import cv2
import numpy as np
from pathlib import Path


def calibrate_charuco(images_dir, squares_x, squares_y, square_size, marker_size,
                      aruco_dict=cv2.aruco.DICT_5X5_100):

    images_dir = Path(images_dir)

    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)

    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_size,
        marker_size,
        dictionary
    )

    detector = cv2.aruco.CharucoDetector(board)

    image_paths = sorted([
        *images_dir.glob("*.jpg"),
        *images_dir.glob("*.jpeg"),
        *images_dir.glob("*.png")
    ])

    m = len(image_paths)

    if not image_paths:
        raise RuntimeError(f"В папке нет изображений: {images_dir}")

    object_points_all = []
    image_points_all = []

    image_size = None

    for image_path in image_paths:

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        current_size = (gray.shape[1], gray.shape[0])

        if image_size is None:
            image_size = current_size

        elif current_size != image_size:
            continue

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) < 4:
            continue

        object_points, image_points = board.matchImagePoints(
            charuco_corners,
            charuco_ids
        )

        object_points_all.append(
            object_points.astype(np.float32)
        )

        image_points_all.append(
            image_points.astype(np.float32)
        )

    n = len(object_points_all)

    if len(object_points_all) < 2:
        raise RuntimeError(
            "Недостаточно изображений с распознанным ChArUco-паттерном."
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objectPoints=object_points_all,
        imagePoints=image_points_all,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    return camera_matrix, dist_coeffs, rms, n, m