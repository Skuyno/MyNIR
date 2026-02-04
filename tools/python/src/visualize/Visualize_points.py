import json
import cv2
import os
from pathlib import Path


def visualize_points(
    json_path: str,
    output_dir: str = "overlays",
    max_images: int | None = 50,
    show: bool = False,
):
    """
    Визуализирует points_2d на кадрах из датасета.

    :param json_path: путь к dataset.json
    :param output_dir: папка, куда сохранять картинки с точками
    :param max_images: максимум кадров для обработки (None = все)
    :param show: показывать ли окна с изображениями (cv2.imshow)
    """

    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Загружено записей: {len(data)}")

    # Базовая папка для картинок — та же, где лежит dataset.json
    base_dir = json_path.parent

    count = 0
    for item in data:
        image_rel_path = item["image"]          # например: "frames/frame_000000.png"
        points_2d = item["points_2d"]

        image_path = base_dir / image_rel_path
        if not image_path.exists():
            print(f"⚠ Картинка не найдена: {image_path}")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"⚠ Не удалось прочитать: {image_path}")
            continue

        h, w, _ = img.shape

        # Рисуем точки
        for (u, v) in points_2d:
            x = int(round(u))
            y = int(round(v))

            # На всякий случай проверим, что точка внутри изображения
            if 0 <= x < w and 0 <= y < h:
                # маленький зелёный кружок
                cv2.circle(img, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

        # Имя файла для сохранения
        out_name = Path(image_rel_path).name  # только имя, без "frames/"
        out_path = output_dir / out_name

        cv2.imwrite(str(out_path), img)

        if show:
            cv2.imshow("points_2d overlay", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC - выйти
                break

        count += 1
        if max_images is not None and count >= max_images:
            break

    if show:
        cv2.destroyAllWindows()

    print(f"Готово! Сохранено {count} изображений с точками в {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Визуализация points_2d на кадрах датасета")
    parser.add_argument(
        "json_path",
        help="Путь к dataset.json (например, output_dataset/dataset.json)",
    )
    parser.add_argument(
        "--out",
        default="overlays",
        help="Папка для сохранения изображений с точками (по умолчанию overlays)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=50,
        help="Максимальное число изображений для обработки (по умолчанию 50)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Показывать ли изображения в окне",
    )

    args = parser.parse_args()

    visualize_points(
        json_path=args.json_path,
        output_dir=args.out,
        max_images=args.max_images,
        show=args.show,
    )
