import cv2
import numpy as np

from pathlib import Path



def grid_is_valid(centers, object_points_template):

    image_points = centers.reshape(-1, 2)

    object_points = object_points_template[:, :2]

    homography, _ = cv2.findHomography(
        object_points,
        image_points
    )

    if homography is None:
        return False

    projected = cv2.perspectiveTransform(
        object_points.reshape(-1, 1, 2),
        homography
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        projected - image_points,
        axis=1
    )

    homography_rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    points = centers.reshape(-1, 2)

    nearest_distances = []

    for point in points:

        distances = np.linalg.norm(
            points - point,
            axis=1
        )

        distances = distances[
            distances > 0
        ]

        nearest_distances.append(
            np.min(distances)
        )

    point_spacing = np.median(
        nearest_distances
    )

    return homography_rmse < point_spacing * 0.15




def calibrate_circles(images_dir, pattern_size, point_distance):

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

    object_points_template *= point_distance

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
            f"В папке нет изображений: {images_dir}"
        )

    object_points_all = []
    image_points_all = []

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
            '''
            print(
                f"Пропущено из-за другого размера: "
                f"{image_path.name}"
            )
            '''
            continue

        found, centers = cv2.findCirclesGrid(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=blob_detector
        )

        used_clustering = False

        if not found:

            found, centers = cv2.findCirclesGrid(
                gray,
                pattern_size,
                flags=(
                    cv2.CALIB_CB_SYMMETRIC_GRID
                    | cv2.CALIB_CB_CLUSTERING
                ),
                blobDetector=blob_detector
            )

            used_clustering = True


        if found and not grid_is_valid(
            centers,
            object_points_template
        ):
            '''
            print(
                f"Неверный порядок точек: "
                f"{image_path.name}"
            )
            '''
            found = False


        if not found:
            '''
            print(
                f"Circles Grid не найден: "
                f"{image_path.name}"
            )
            '''
            continue

        '''
        print(
            f"OK: {image_path.name} — "
            f"{len(centers)} точек"
        )
        '''

        object_points_all.append(
            object_points_template.copy()
        )

        image_points_all.append(
            centers.astype(
                np.float32
            )
        )

    n = len(
        object_points_all
    )

    if n < 2:
        raise RuntimeError(
            "Недостаточно изображений "
            "с распознанным Symmetric Circles Grid."
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objectPoints=object_points_all,
        imagePoints=image_points_all,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    return (
        camera_matrix,
        dist_coeffs,
        rms,
        n,
        m
    )