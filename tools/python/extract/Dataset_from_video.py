import cv2
import json
from pathlib import Path

import mediapipe as mp
import numpy as np

LANDMARK_COUNT = 478


def normalize_points(
    points: np.ndarray,
    nose_index: int = 1,
    left_eye_index: int = 33,
    right_eye_index: int = 263,
) -> np.ndarray:
    center = points[nose_index]
    left_eye = points[left_eye_index]
    right_eye = points[right_eye_index]
    eye_distance = np.linalg.norm(left_eye - right_eye)
    if eye_distance <= 1e-6:
        return np.zeros_like(points)
    return (points - center) / eye_distance


def ema_smooth(points: np.ndarray, previous: np.ndarray | None, alpha: float) -> np.ndarray:
    if previous is None:
        return points
    return alpha * points + (1.0 - alpha) * previous

def build_dataset(
    video_name: str,
    output_dir: str | None = None,
    output_name: str = "dataset.json",
    smoothing_alpha: float = 0.7,
):
    video_name = Path(video_name)
    repo_root = Path(__file__).resolve().parents[3]

    if not video_name.is_absolute():
        video_name = repo_root / "Data" / video_name
    if output_dir is None:
        output_dir = repo_root / "data"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_name))
    if not cap.isOpened():
        raise RuntimeError(f"Can't open the video: {video_name}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_id = 0
    previous_smoothed = None

    print("Script start")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            pts3d = []

            for lm in face_landmarks.landmark:
                x_norm, y_norm, z_norm = lm.x, lm.y, lm.z
                pts3d.append([float(x_norm), float(y_norm), float(z_norm)])

            points = np.array(pts3d, dtype=np.float32)
            normalized = normalize_points(points)
            smoothed = ema_smooth(normalized, previous_smoothed, smoothing_alpha)
            previous_smoothed = smoothed

            frames.append(
                {
                    "frame_id": int(frame_id),
                    "valid": True,
                    "points_3d": smoothed.tolist(),
                }
            )
        else:
            frames.append(
                {
                    "frame_id": int(frame_id),
                    "valid": False,
                    "points_3d": np.zeros((LANDMARK_COUNT, 3), dtype=np.float32).tolist(),
                }
            )

        frame_id += 1
            
        if frame_id % 50 == 0:
            print(f"Обратно кадров: {frame_id}")
    cap.release()
    face_mesh.close()

    dataset = {
        "meta": {
            "fps": float(fps) if fps else 0.0,
            "landmarks": LANDMARK_COUNT,
        },
        "frames": frames,
    }

    json_path = output_dir / output_name
    with open(json_path, "w", encoding = "utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    valid_count = sum(1 for frame in frames if frame["valid"])
    print(f"Готово! Сохранено {valid_count} кадров с лицом.")
    print(f"Описания: {json_path}")
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Построение датасета с 478 лицевыми точками")
    parser.add_argument("video", help="Путь к видеофайлу")
    parser.add_argument(
        "--out",
        help="Папка для сохранения датасета (по умолчанию output_dataset)",
    )
    parser.add_argument(
        "--output_name",
        default="dataset.json",
        help="Имя JSON файла с результатами (по умолчанию dataset.json)",
    )
    parser.add_argument(
        "--smoothing_alpha",
        type=float,
        default=0.7,
        help="EMA коэффициент сглаживания (0..1, больше = меньше сглаживания)",
    )

    args = parser.parse_args()
    build_dataset(
        args.video,
        args.out,
        output_name=args.output_name,
        smoothing_alpha=args.smoothing_alpha,
    )