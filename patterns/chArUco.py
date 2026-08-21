import cv2
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "generated"
OUTPUT_DIR.mkdir(exist_ok=True)

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_5X5_100
)

board = cv2.aruco.CharucoBoard(
    (5, 7),
    30.0,   # square size
    20.0,   # marker size
    dictionary
)

image = board.generateImage(
    (1400, 2000),
    marginSize=100,
    borderBits=1
)

output_path = OUTPUT_DIR / "charuco.png"
cv2.imwrite(str(output_path), image)

print(f"Saved: {output_path}")