import cv2
import numpy as np


def undistort_frame(frame, camera_matr, dist_coef):

    h, w = frame.shape[:2]

    new_camera_matr, roi = cv2.getOptimalNewCameraMatrix(
        camera_matr, dist_coef, (w, h), 1, (w, h)
    )

    undistorted_frame = cv2.undistort(
        frame, camera_matr, dist_coef, None, new_camera_matr
    )

    return undistorted_frame, new_camera_matr


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


def mask_red(frame, h_margin=10, s_margin=40, v_margin=40, search_radius=50):
    """Искать красный цвет в радиусе search_radius пикселей от каждого клика."""
    return mask_color_manual(
        frame, 4, "Red", h_margin, s_margin, v_margin,
        search_radius=search_radius
    )


def mask_blue(frame, h_margin=10, s_margin=40, v_margin=40, search_radius=40):
    """Искать синий цвет в радиусе search_radius пикселей от каждого клика."""
    return mask_color_manual(
        frame, 3, "Blue", h_margin, s_margin, v_margin,
        search_radius=search_radius
    )


def mask_color_manual(frame, count, color, h_margin=10, s_margin=40, v_margin=40,
                      search_radius=None):
    """Выбрать точки внутри кругов и расширить диапазон HSV."""
    if any(value < 0 for value in (h_margin, s_margin, v_margin)):
        raise ValueError("Запас HSV должен быть неотрицательным")
    if search_radius is not None:
        if not np.isfinite(search_radius) or search_radius <= 0:
            raise ValueError("Радиус поиска должен быть положительным")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    points = []
    h, w = frame.shape[:2]
    scale = min(1.0, 1200 / w, 800 / h)
    pw, ph = max(1, round(w * scale)), max(1, round(h * scale))
    preview = cv2.resize(frame, (pw, ph))
    name = f"{color}: select {count} points"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < count:
            if 0 <= x < pw and 0 <= y < ph:
                points.append((min(w - 1, int(x * w / pw)),
                               min(h - 1, int(y * h / ph))))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setMouseCallback(name, on_mouse)
        while True:
            display = preview.copy()
            for i, (x, y) in enumerate(points, 1):
                pos = (round(x * pw / w), round(y * ph / h))
                if search_radius is not None:
                    axes = (max(1, round(search_radius * pw / w)),
                            max(1, round(search_radius * ph / h)))
                    cv2.ellipse(display, pos, axes, 0, 0, 360, (0, 255, 255), 1)
                cv2.circle(display, pos, 5, (0, 255, 0), 2)
                cv2.putText(display, str(i), pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for i, label in enumerate((
                f"{len(points)}/{count}  LMB: add  RMB: undo",
                "Enter: confirm  Esc: cancel"
            )):
                pos = (8, 20 + 22 * i)
                cv2.putText(display, label, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(display, label, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow(name, display)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError(f"Выбор точек {color} отменён")
            if key in (10, 13) and len(points) == count:
                break
    finally:
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass

    samples = np.array([hsv[y, x] for x, y in points], dtype=np.int32)
    # Оттенок циклический: выбираем кратчайшую дугу между образцами.
    hues = np.sort(samples[:, 0])
    gaps = np.diff(np.r_[hues, hues[0] + 180])
    gap_index = int(np.argmax(gaps))
    start = int(hues[(gap_index + 1) % len(hues)])
    span = 180 - int(gaps[gap_index])
    hue_mask = ((hsv[:, :, 0].astype(np.int32) - start + h_margin) % 180
                <= span + 2 * h_margin)
    lower = np.maximum(samples[:, 1:].min(axis=0) - [s_margin, v_margin], 0)
    upper = np.minimum(samples[:, 1:].max(axis=0) + [s_margin, v_margin], 255)
    sv_mask = np.all((hsv[:, :, 1:] >= lower) & (hsv[:, :, 1:] <= upper), axis=2)
    mask = hue_mask & sv_mask
    if search_radius is not None:
        # Координаты и радиус относятся к исходному кадру, не к предпросмотру.
        yy, xx = np.ogrid[:h, :w]
        search_area = np.zeros((h, w), dtype=bool)
        for x, y in points:
            search_area |= (xx - x) ** 2 + (yy - y) ** 2 <= search_radius ** 2
        mask &= search_area
    return mask.astype(np.uint8) * 255


def find_ellipses(mask, min_area=100, max_area=30000):

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

        '''print(f"Площадь эллипса: {area:.1f} px²")'''

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

        '''
        print(
            f"Круг {i + 1}: "
            f"d = {diameter_px:.2f} px, "
            f"масштаб = {scale:.4f} px/mm"
        )
        '''

    mean_diameter_px = np.mean(diameters_px)

    px_per_mm = mean_diameter_px / real_diameter_mm
    mm_per_px = 1 / px_per_mm
    '''
    print(f"\nСредний диаметр: {mean_diameter_px:.2f} px")
    print(f"Масштаб: {px_per_mm:.4f} px/mm")
    print(f"Масштаб: {mm_per_px:.5f} mm/px")
    '''

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

    distances_mm = sorted(distances_mm)

    d_min = distances_mm[0]
    d_mean = distances_mm[1]
    d_max = distances_mm[2]

    #print(f"D_min:  {d_min:.2f} mm   (должно быть 18)")
    #print(f"D_mean: {d_mean:.2f} mm   (должно быть 24)")
    #print(f"D_max:  {d_max:.2f} mm   (должно быть 30)")

    error = (
        (
            abs(18 - d_min) / 18
            + abs(24 - d_mean) / 24
            + abs(30 - d_max) / 30
        )
        / 3
        * 100
    )

    return error

def show_frame(name, frame):
    """Показать отладочный кадр; любая клавиша или закрытие окна — дальше."""
    h, w = frame.shape[:2]
    scale = min(1.0, 1200 / w, 800 / h)
    preview = cv2.resize(
        frame, (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_NEAREST if frame.ndim == 2 else cv2.INTER_AREA
    )
    cv2.imshow(name, preview)
    try:
        while cv2.waitKey(20) == -1:
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass


def test_one_combination(CAMERA_MATR, DIST_COEF, frame):

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

    MARKER_WIDTH = 40.0
    MARKER_HEIGHT = 65.0



    

    #show_frame("Ellipses", ELLIPSES_FRAME)
    #show_frame("Perspective corrected", CORRECTED_FRAME)

    SCALE = 10

    CORRECTED_FRAME, HOMOGRAPHY = perspective_correction(UNDISTORTED_FRAME, CENTERS, MARKER_WIDTH, MARKER_HEIGHT)

    BLUE_MASK = mask_blue(CORRECTED_FRAME)
        
    ELLIPSES_BLUE = find_ellipses(BLUE_MASK)

    if len(ELLIPSES_BLUE) != 3:
        raise RuntimeError(
            f"Синих кругов найдено: {len(ELLIPSES_BLUE)}, ожидалось 3"
        )
    
    ELLIPSES_FRAME_BLUE = draw_ellipses(CORRECTED_FRAME, ELLIPSES_BLUE)
    
    CENTERS_BLUE = get_ellipse_centers(ELLIPSES_BLUE)


    #print(f"PX_PER_MM = {PX_PER_MM:.3f}")
    #print(f"MM_PER_PX = {MM_PER_PX:.5f}")
    # show_frame("Ellipses blue", ELLIPSES_FRAME_BLUE)
    

    MM_PER_PX = 1 / SCALE

    DISTANCES = calculate_distances(
        ELLIPSES_BLUE,
        MM_PER_PX
    )

    for i, j, distance_px, distance_mm in DISTANCES:
        '''
        print(
            f"D{i}{j}: "
            f"{distance_px:.2f} px = "
            f"{distance_mm:.2f} mm"
        )
        '''

    ERROR = calculate_error(DISTANCES)



    return ERROR