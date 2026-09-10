import cv2

from pathlib import Path
from datetime import datetime


SAVE_DIR = Path("data/webcam_12")

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


camera = cv2.VideoCapture(0)

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(
    "Resolution:",
    int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "x",
    int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

print(
    "Resolution:",
    int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "x",
    int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
)


if not camera.isOpened():
    raise RuntimeError(
        "Не удалось открыть веб-камеру."
    )


while True:

    ret, frame = camera.read()

    print(
        "Frame:",
        frame.shape[1],
        "x",
        frame.shape[0]
    )

    if not ret:
        print("Не удалось получить кадр.")
        break

    cv2.imshow(
        "Web camera",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # Пробел
    if key == 32:

        filename = "test.jpg"

        # datetime.now().strftime("%Y%m%d_%H%M%S_%f") + 

        save_path = (
            SAVE_DIR
            / filename
        )

        cv2.imwrite(
            str(save_path),
            frame
        )

        print(
            f"Сохранено: {save_path}"
        )

    # Esc
    elif key == 27:
        break


camera.release()
cv2.destroyAllWindows()
