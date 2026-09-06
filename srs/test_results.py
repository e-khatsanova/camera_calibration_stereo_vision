'''
План проверки полученных параметров такой

исходный кадр
    ↓
undistort(K, dist)
    ↓
perspective correction / homography
    ↓
фронтальный вид мишени
    ↓
по радиусам R1, R2, R3 определяем масштаб mm/px
    ↓
измеряем расстояния между центрами
    ↓
сравниваем с реальными D12, D13, D23
'''

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("test")
CALIB_RESULTS_DIR = Path("calib_results")

TEST_VIDEO = Path("data/iphone/test.MOV")
CUT = 5  # Примерно середина

CAMERAS = ["iphone"]
TARGETS = ["a4", "tab"]
PATTERNS = ["chess", "charuco", "circle", "acircle"]
N_VALUES = range(10, 31)

def main():
    results = []

    for camera_i in CAMERAS:
        for target_j in TARGETS:
            for pattern_k in PATTERNS:
                for n_g in N_VALUES:

                    print(
                        f"{camera_i}, {target_j}, "
                        f"{pattern_k}, N={n_g}"
                    )

                    try:
                        error_this_combination = test_one_combination(
                            camera_i,
                            target_j,
                            pattern_k,
                            n_g
                        )

                    except Exception as error:
                        print(f"Ошибка: {error}")
                        continue

                    results.append([
                        camera_i,
                        target_j,
                        pattern_k,
                        n_g,
                        error_this_combination
                    ])

    save_results(results)

    RESULT_FILE = Path("test/calibration_test_results.csv")

    plot_results(RESULT_FILE, "a4")
    plot_results(RESULT_FILE, "tab")
















def save_results(results):

    df = pd.DataFrame(
        results,
        columns=[
            "camera",
            "target",
            "pattern",
            "n",
            "error"
        ]
    )

    df.to_csv(
        "test/calibration_test_results.csv",
        index=False
    )

    print("\nРезультаты сохранены:")
    print(df)


def plot_results(file, target):

    df = pd.read_csv(file)

    df = df[df["target"] == target]

    plt.figure(figsize=(12, 7))

    for (camera, pattern), group in df.groupby(["camera", "pattern"]):

        group = group.sort_values("n")

        plt.plot(
            group["n"],
            group["error"],
            marker="o",
            label=f"{camera} - {pattern}"
        )

    plt.xlabel("N")
    plt.ylabel("Error")
    plt.title(f"Calibration error — {target}")

    plt.xticks(sorted(df["n"].unique()))
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()



























def get_test_frame():

    cap = cv2.VideoCapture(str(TEST_VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {TEST_VIDEO}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, CUT)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"Не удалось получить кадр {CUT}")

    return frame


def take_parameters(camera, target, pattern, n):

    calib_file = CALIB_RESULTS_DIR / f"{camera}_{pattern}_{target}_n{n}.npz"

    if not calib_file.exists():
        raise RuntimeError(f"Файл калибровки не найден: {calib_file}")

    data = np.load(calib_file)
    rms = float(data["median_rms"])
    camera_matr = data["median_camera_matrix"]
    dist_coef = data["median_dist_coeffs"]

    return rms, camera_matr, dist_coef


def save_test_frame(frame):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_file = OUTPUT_DIR / f"frame_{CUT}.jpg"
    cv2.imwrite(str(frame_file), frame)
    print(f"Тестовый кадр сохранён: {frame_file}")
    print(f"Размер: {frame.shape[1]} x {frame.shape[0]}")


def undistort_frame(frame, camera_matr, dist_coef):

    h, w = frame.shape[:2]

    new_camera_matr, roi = cv2.getOptimalNewCameraMatrix(
        camera_matr, dist_coef, (w, h), 1, (w, h)
    )

    undistorted_frame = cv2.undistort(
        frame, camera_matr, dist_coef, None, new_camera_matr
    )

    return undistorted_frame, new_camera_matr



def show_frame(name, frame, scale=1):

    h, w = frame.shape[:2]

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_frame = cv2.resize(frame, (new_w, new_h))

    cv2.imshow(name, resized_frame)


def sort_corners(corners):

    sorted_corners = np.zeros((4, 2), dtype=np.float32)

    point_sum = corners.sum(axis=1)
    point_diff = np.diff(corners, axis=1).reshape(-1)

    sorted_corners[0] = corners[np.argmin(point_sum)]   # Левый верхний
    sorted_corners[1] = corners[np.argmin(point_diff)] # Правый верхний
    sorted_corners[2] = corners[np.argmax(point_sum)]   # Правый нижний
    sorted_corners[3] = corners[np.argmax(point_diff)]  # Левый нижний

    return sorted_corners





def binarize_frame(frame, threshold=100):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, threshold, 255, cv2.THRESH_BINARY
    )

    return binary



def mask_top_bottom(frame, top=20, bottom=15):

    result = frame.copy()

    h, w = result.shape[:2]

    top_y = int(h * top / 100)
    bottom_y = int(h * (100 - bottom) / 100)

    cv2.rectangle(result, (0, 0), (w, top_y), 255, -1)
    cv2.rectangle(result, (0, bottom_y), (w, h), 255, -1)

    return result


def mask_red(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_1 = cv2.inRange(
        hsv,
        np.array([0, 80, 50]),
        np.array([10, 255, 255])
    )

    mask_2 = cv2.inRange(
        hsv,
        np.array([170, 80, 50]),
        np.array([180, 255, 255])
    )

    red_mask = cv2.bitwise_or(mask_1, mask_2)

    return red_mask

def mask_blue(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(
        hsv,
        np.array([90, 80, 50]),
        np.array([130, 255, 255])
    )

    return blue_mask


def find_ellipses(mask, min_area=100, max_area=10000):

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    ellipses = []

    for contour in contours:

        if len(contour) < 5:
            continue

        ellipse = cv2.fitEllipse(contour)

        a, b = ellipse[1]
        area = np.pi * a * b / 4

        print(f"Площадь эллипса: {area:.1f} px²")

        if area < min_area or area > max_area:
            continue

        ellipses.append(ellipse)

    return ellipses


def draw_ellipses(frame, ellipses):

    result = frame.copy()

    for ellipse in ellipses:
        cv2.ellipse(result, ellipse, (0, 255, 0), 3)

    return result



def get_ellipse_centers(ellipses):

    centers = []

    for ellipse in ellipses:
        x, y = ellipse[0]
        centers.append([x, y])

    return np.array(centers, dtype=np.float32)




def order_points(points):

    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(sums)]   # top-left
    ordered[2] = points[np.argmax(sums)]   # bottom-right

    ordered[1] = points[np.argmin(diffs)]  # top-right
    ordered[3] = points[np.argmax(diffs)]  # bottom-left

    return ordered



def perspective_correction(frame, points, width_mm, height_mm, scale=10):

    src_points = order_points(points)

    width = int(width_mm * scale)
    height = int(height_mm * scale)

    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    homography = cv2.getPerspectiveTransform(
        src_points,
        dst_points
    )

    corrected = cv2.warpPerspective(
        frame,
        homography,
        (width, height)
    )

    return corrected, homography




def calibrate_scale(ellipses, real_diameter_mm=5.0):

    diameters_px = []

    for i, ellipse in enumerate(ellipses):

        a, b = ellipse[1]

        diameter_px = np.sqrt(a * b)
        scale = diameter_px / real_diameter_mm

        diameters_px.append(diameter_px)

        print(
            f"Круг {i + 1}: "
            f"d = {diameter_px:.2f} px, "
            f"масштаб = {scale:.4f} px/mm"
        )

    mean_diameter_px = np.mean(diameters_px)

    px_per_mm = mean_diameter_px / real_diameter_mm
    mm_per_px = 1 / px_per_mm

    print(f"\nСредний диаметр: {mean_diameter_px:.2f} px")
    print(f"Масштаб: {px_per_mm:.4f} px/mm")
    print(f"Масштаб: {mm_per_px:.5f} mm/px")

    return px_per_mm, mm_per_px




def calculate_distances(ellipses, mm_per_px):

    centers = []

    for ellipse in ellipses:
        x, y = ellipse[0]
        centers.append(np.array([x, y]))

    distances = []

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):

            distance_px = np.linalg.norm(
                centers[i] - centers[j]
            )

            distance_mm = distance_px * mm_per_px

            distances.append(
                (i + 1, j + 1, distance_px, distance_mm)
            )

    return distances



def calculate_error(distances):

    distances_mm = [
        distance[3] for distance in distances
    ]

    d_min = min(distances_mm)
    d_max = max(distances_mm)
    d_mean = sorted(distances_mm)[1]

    error = ((abs(18 - d_min) / 18) + (abs(24 - d_mean) / 24) + (abs(30 - d_max) / 30) )*100

    print(f"D_min:  {d_min:.2f} mm")
    print(f"D_mean: {d_mean:.2f} mm")
    print(f"D_max:  {d_max:.2f} mm")
    print(f"Ошибка: {error:.6f} %")

    return error


def test_one_combination(CAMERA, TARGET, PATTERN, N):

    frame = get_test_frame()
    save_test_frame(frame)

    RMS, CAMERA_MATR, DIST_COEF = take_parameters(CAMERA, TARGET, PATTERN, N)

    print("RMS:", RMS)
    print("CAMERA_MATR:\n", CAMERA_MATR)
    print("DIST_COEF:", DIST_COEF)

    UNDISTORTED_FRAME, NEW_CAMERA_MATR = undistort_frame(
        frame, CAMERA_MATR, DIST_COEF
    )
    
    RED_MASK = mask_red(UNDISTORTED_FRAME)

    ELLIPSES = find_ellipses(RED_MASK)

    if len(ELLIPSES) != 4:
        raise RuntimeError(
            f"Красных маркеров найдено: {len(ELLIPSES)}, ожидалось 4"
        )

    ELLIPSES_FRAME = draw_ellipses(UNDISTORTED_FRAME, ELLIPSES )

    CENTERS = get_ellipse_centers(ELLIPSES)


    MARKER_WIDTH = 39.0
    MARKER_HEIGHT = 64.0


    CORRECTED_FRAME, HOMOGRAPHY = perspective_correction(UNDISTORTED_FRAME, CENTERS, MARKER_WIDTH, MARKER_HEIGHT)

    #show_frame("Ellipses", ELLIPSES_FRAME)
    #show_frame("Perspective corrected", CORRECTED_FRAME)



    BLUE_MASK = mask_blue(CORRECTED_FRAME)
    
    ELLIPSES_BLUE = find_ellipses(BLUE_MASK)

    if len(ELLIPSES_BLUE) != 3:
        raise RuntimeError(
            f"Синих кругов найдено: {len(ELLIPSES_BLUE)}, ожидалось 3"
        )
    
    ELLIPSES_FRAME_BLUE = draw_ellipses(CORRECTED_FRAME, ELLIPSES_BLUE)
    
    CENTERS_BLUE = get_ellipse_centers(ELLIPSES_BLUE)

    PX_PER_MM, MM_PER_PX = calibrate_scale(
        ELLIPSES_BLUE,
        real_diameter_mm=5.0
    )

    # show_frame("Ellipses blue", ELLIPSES_FRAME_BLUE)


    DISTANCES = calculate_distances(ELLIPSES_BLUE, MM_PER_PX)
    for i, j, distance_px, distance_mm in DISTANCES:

        print(
            f"D{i}{j}: "
            f"{distance_px:.2f} px = "
            f"{distance_mm:.2f} mm"
        )

    ERROR = calculate_error(DISTANCES)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ERROR














if __name__ == "__main__":
    main()