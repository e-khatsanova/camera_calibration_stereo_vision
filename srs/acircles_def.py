import cv2
import numpy as np

from pathlib import Path


def calibrate_acircles(images_dir, pattern_size, point_distance):

    images_dir = Path(images_dir)

    object_points_template = np.zeros(
        (
            pattern_size[0] * pattern_size[1],
            3
        ),
        dtype=np.float32
    )

    point_index = 0

    for row in range(pattern_size[1]):

        for column in range(pattern_size[0]):

            object_points_template[
                point_index,
                0
            ] = (
                2 * column
                + row % 2
            ) * point_distance

            object_points_template[
                point_index,
                1
            ] = (
                row
                * point_distance
            )

            point_index += 1

    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 50000

    params.filterByCircularity = True
    params.minCircularity = 0.6

    params.filterByConvexity = True
    params.minConvexity = 0.7

    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    blob_detector = cv2.SimpleBlobDetector_create(
        params
    )

    image_paths = sorted([
        *images_dir.glob("*.jpg"),
        *images_dir.glob("*.jpeg"),
        *images_dir.glob("*.png"),
    ])

    m = len(image_paths)

    if not image_paths:
        raise RuntimeError(
            f"В папке нет изображений: "
            f"{images_dir}"
        )

    object_points_all = []
    image_points_all = []

    image_size = None

    for image_path in image_paths:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
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

        found, centers = cv2.findCirclesGrid(
            gray,
            pattern_size,
            flags=(
                cv2.CALIB_CB_ASYMMETRIC_GRID
                | cv2.CALIB_CB_CLUSTERING
            ),
            blobDetector=blob_detector
        )

        if not found:
            continue


        object_points_all.append(
            object_points_template.copy()
        )

        image_points_all.append(
            centers.astype(np.float32)
        )

    n = len(object_points_all)

    if len(object_points_all) < 2:

        raise RuntimeError(
            "Недостаточно валидных кадров."
        )

    (
        rms,
        camera_matrix,
        dist_coeffs,
        _,
        _
    ) = cv2.calibrateCamera(
        objectPoints=object_points_all,
        imagePoints=image_points_all,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    return camera_matrix, dist_coeffs, rms, n, m