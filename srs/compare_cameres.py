import cv2


from charuco_def import calibrate_charuco
from acircles_def import calibrate_acircles
from circles_def import calibrate_circles
from chess_def import calibrate_chess
from test_def import test_one_combination



camera_matrix_i_char, dist_coeffs_i_char, rms_i_char, n_i_char, m_i_char = calibrate_charuco(
    images_dir="data/iphone_12/charuco",
    squares_x=5,
    squares_y=7,
    square_size=30.0,
    marker_size=20.0
)

camera_matrix_i_acircles, dist_coeffs_i_acircles, rms_i_acircles, n_i_acircles, m_i_acircles = calibrate_acircles(
    images_dir=r"data/iphone_12/acircles",
    pattern_size=(4, 9),
    point_distance=13.5
)

camera_matrix_i_circles, dist_coeffs_i_circles, rms_i_circles, n_i_circles, m_i_circles = calibrate_circles(
    images_dir=r"data/iphone_12/circles",
    pattern_size=(6, 9),
    point_distance=13.5
)


camera_matrix_i_chess, dist_coeffs_i_chess, rms_i_chess, n_i_chess, m_i_chess = calibrate_chess(
    images_dir=r"data/iphone_12/chess",
    pattern_size=(8, 5),
    square_size=13.5
)



test_frame_i =cv2.imread(str("data/iphone_12/test.JPG"))    
error_i_char = test_one_combination(camera_matrix_i_char, dist_coeffs_i_char, test_frame_i)    
error_i_acircles = test_one_combination(camera_matrix_i_acircles, dist_coeffs_i_acircles, test_frame_i)
error_i_circles = test_one_combination(camera_matrix_i_circles, dist_coeffs_i_circles, test_frame_i)
error_i_chess = test_one_combination(camera_matrix_i_chess, dist_coeffs_i_chess, test_frame_i)



results_i = {
    "Chessboard": (rms_i_chess, n_i_chess, m_i_chess, error_i_chess),
    "ChArUco": (rms_i_char, n_i_char, m_i_char, error_i_char),
    "Circles": (rms_i_circles, n_i_circles, m_i_circles, error_i_circles),
    "A-Circles": (rms_i_acircles, n_i_acircles, m_i_acircles, error_i_acircles),
}

print()
print(f"{'Pattern':<14} | {'iPhone 17':<20}")
print("-" * 37)

for pattern, (rms, n, m, error) in results_i.items():
    print(
        f"{pattern:<14} | "
        f"{rms:.2f}, {n}/{m}, Error: {error:.2f}%"
    )





camera_matrix_w_acircles, dist_coeffs_w_acircles, rms_w_acircles, n_w_acircles, m_w_acircles = calibrate_acircles(
    images_dir=r"data/webcam_12/acircles",
    pattern_size=(4, 9),
    point_distance=13.5
)

camera_matrix_w_circles, dist_coeffs_w_circles, rms_w_circles, n_w_circles, m_w_circles = calibrate_circles(
    images_dir=r"data/webcam_12/circles",
    pattern_size=(6, 9),
    point_distance=13.5
)


camera_matrix_w_chess, dist_coeffs_w_chess, rms_w_chess, n_w_chess, m_w_chess = calibrate_chess(
    images_dir=r"data/webcam_12/chess",
    pattern_size=(8, 5),
    square_size=13.5
)


test_frame_w =cv2.imread(str("data/webcam_12/test.JPG"))    
# error_w_char = test_one_combination(camera_matrix_w_char, dist_coeffs_w_char, test_frame_w)  

error_w_acircles = test_one_combination(camera_matrix_w_acircles, dist_coeffs_w_acircles, test_frame_w)
error_w_circles = test_one_combination(camera_matrix_w_circles, dist_coeffs_w_circles, test_frame_w)
error_w_chess = test_one_combination(camera_matrix_w_chess, dist_coeffs_w_chess, test_frame_w)



results_w = {
    "Chessboard": (rms_w_chess, n_w_chess, m_w_chess, error_w_chess),
    "ChArUco": (0, 0, 0, 0),
    "Circles": (rms_w_circles, n_w_circles, m_w_circles, error_w_circles),
    "A-Circles": (rms_w_acircles, n_w_acircles, m_w_acircles, error_w_acircles),
}

print()
print(f"{'Pattern':<14} | {'webcam':<20}")
print("-" * 37)

for pattern, (rms, n, m, error) in results_w.items():
    print(
        f"{pattern:<14} | "
        f"{rms:.2f}, {n}/{m}, Error: {error:.2f}%"
    )









camera_matrix_g1_acircles, dist_coeffs_g1_acircles, rms_g1_acircles, n_g1_acircles, m_g1_acircles = calibrate_acircles(
    images_dir=r"data/GoPro_rus_12/acircles",
    pattern_size=(4, 9),
    point_distance=13.5
)

camera_matrix_g1_circles, dist_coeffs_g1_circles, rms_g1_circles, n_g1_circles, m_g1_circles = calibrate_circles(
    images_dir=r"data/GoPro_rus_12/circles",
    pattern_size=(6, 9),
    point_distance=13.5
)

camera_matrix_g1_chess, dist_coeffs_g1_chess, rms_g1_chess, n_g1_chess, m_g1_chess = calibrate_chess(
    images_dir=r"data/GoPro_rus_12/chess",
    pattern_size=(8, 5),
    square_size=13.5
)

camera_matrix_g1_char, dist_coeffs_g1_char, rms_g1_char, n_g1_char, m_g1_char = calibrate_charuco(
    images_dir="data/GoPro_rus_12/charuco",
    squares_x=5,
    squares_y=7,
    square_size=30.0,
    marker_size=20.0
)



test_frame_g1 =cv2.imread(str("data/GoPro_rus_12/test.JPG"))    

error_g1_acircles = test_one_combination(camera_matrix_g1_acircles, dist_coeffs_g1_acircles, test_frame_g1)
error_g1_circles = test_one_combination(camera_matrix_g1_circles, dist_coeffs_g1_circles, test_frame_g1)
error_g1_chess = test_one_combination(camera_matrix_g1_chess, dist_coeffs_g1_chess, test_frame_g1)
error_g1_char = test_one_combination(camera_matrix_g1_char, dist_coeffs_g1_char, test_frame_g1)


results_g1 = {
    "Chessboard": (rms_g1_chess, n_g1_chess, m_g1_chess, error_g1_chess),
    "ChArUco": (rms_g1_char, n_g1_char, m_g1_char, error_g1_char),
    "Circles": (rms_g1_circles, n_g1_circles, m_g1_circles, error_g1_circles),
    "A-Circles": (rms_g1_acircles, n_g1_acircles, m_g1_acircles, error_g1_acircles),
}

print()
print(f"{'Pattern':<14} | {'GoPro rus':<20}")
print("-" * 37)

for pattern, (rms, n, m, error) in results_g1.items():
    print(
        f"{pattern:<14} | "
        f"{rms:.2f}, {n}/{m}, Error: {error:.2f}%"
    )


camera_matrix_g2_acircles, dist_coeffs_g2_acircles, rms_g2_acircles, n_g2_acircles, m_g2_acircles = calibrate_acircles(
    images_dir=r"data/GoPro_papa_12/acircles",
    pattern_size=(4, 9),
    point_distance=13.5
)

camera_matrix_g2_circles, dist_coeffs_g2_circles, rms_g2_circles, n_g2_circles, m_g2_circles = calibrate_circles(
    images_dir=r"data/GoPro_papa_12/circles",
    pattern_size=(6, 9),
    point_distance=13.5
)

camera_matrix_g2_chess, dist_coeffs_g2_chess, rms_g2_chess, n_g2_chess, m_g2_chess = calibrate_chess(
    images_dir=r"data/GoPro_papa_12/chess",
    pattern_size=(8, 5),
    square_size=13.5
)

camera_matrix_g2_char, dist_coeffs_g2_char, rms_g2_char, n_g2_char, m_g2_char = calibrate_charuco(
    images_dir="data/GoPro_papa_12/charuco",
    squares_x=5,
    squares_y=7,
    square_size=30.0,
    marker_size=20.0
)



test_frame_g2 =cv2.imread(str("data/GoPro_papa_12/test.JPG"))    

error_g2_acircles = test_one_combination(camera_matrix_g2_acircles, dist_coeffs_g2_acircles, test_frame_g2)
# error_g2_circles = test_one_combination(camera_matrix_g2_circles, dist_coeffs_g2_circles, test_frame_g2)
error_g2_chess = test_one_combination(camera_matrix_g2_chess, dist_coeffs_g2_chess, test_frame_g2)
error_g2_char = test_one_combination(camera_matrix_g2_char, dist_coeffs_g2_char, test_frame_g2)


results_g2 = {
    "Chessboard": (rms_g2_chess, n_g2_chess, m_g2_chess, error_g2_chess),
    "ChArUco": (rms_g2_char, n_g2_char, m_g2_char, error_g2_char),
    "Circles": (rms_g2_circles, n_g2_circles, m_g2_circles, 0),
    "A-Circles": (rms_g2_acircles, n_g2_acircles, m_g2_acircles, error_g2_acircles),
}

print()
print(f"{'Pattern':<14} | {'GoPro papa':<20}")
print("-" * 37)

for pattern, (rms, n, m, error) in results_g2.items():
    print(
        f"{pattern:<14} | "
        f"{rms:.2f}, {n}/{m}, Error: {error:.2f}%"
    )