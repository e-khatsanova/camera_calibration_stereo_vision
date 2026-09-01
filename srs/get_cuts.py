import cv2

from pathlib import Path


NAME = "circle_a4"

DATA_DIR = Path("data") / "iphone"

VIDEO_PATH = DATA_DIR / f"{NAME}.MOV"
OUTPUT_DIR = DATA_DIR / NAME


SPEED = 0.25
WINDOW_SCALE = 2


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {VIDEO_PATH}")

    # Узнаём FPS исходного видео
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Задержка между кадрами с учётом скорости воспроизведения
    delay = int(1000 / (fps * SPEED))

    saved_count = 0

    print(f"FPS: {fps:.1f}")
    print("Управление:")
    print("g — сохранить кадр")
    print("q — выйти")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Видео закончилось.")
            break

        # Увеличиваем ТОЛЬКО изображение для просмотра
        display_frame = cv2.resize(
            frame,
            None,
            fx=WINDOW_SCALE,
            fy=WINDOW_SCALE,
            interpolation=cv2.INTER_LINEAR
        )

        cv2.imshow("Calibration video", display_frame)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord("g"):
            output_path = OUTPUT_DIR / f"frame_{saved_count:03d}.jpg"

            # Сохраняем исходный frame, а не увеличенный display_frame
            success = cv2.imwrite(str(output_path), frame)

            if success:
                print(f"Сохранён: {output_path}")
                saved_count += 1

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Всего сохранено кадров: {saved_count}")


if __name__ == "__main__":
    main()