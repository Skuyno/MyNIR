import cv2
import mediapipe as mp
import numpy as np
import json
import os
from pathlib import Path

def build_dataset(
    video_path: str,
    output_dir: str = "output_dataset",
    frames_subdir: str = "frames",
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    frames_dir = output_dir / frames_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Can't open the video: {video_path}")
    
    dataset = []
    frame_id = 0

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
            pts2d = []

            for lm in face_landmarks.landmark:
                x_norm, y_norm, z_norm = lm.x, lm.y, lm.z
                pts3d.append([float(x_norm), float(y_norm), float(z_norm)])

                u = float(x_norm * w)
                v = float(y_norm * h)
                pts2d.append([u,v])
            
            img_name = f"frame_{frame_id:06d}.png"
            img_path = frames_dir / img_name
            cv2.imwrite(str(img_path), frame)

            dataset.append(
                {
                    "image": str(Path(frames_subdir)/img_name),
                    "points_3d": pts3d,
                    "points_2d": pts2d,
                    "width": int(w),
                    "height": int(h),
                    "frame_id": int(frame_id)
                }
            )
            
        frame_id += 1
            
        if frame_id % 50 == 0:
            print(f"Обратно кадров: {frame_id}")
    cap.release()
    face_mesh.close()

    json_path = output_dir / "dataset.json"
    with open(json_path, "w", encoding = "utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Готово! Сохранено {len(dataset)} кадров с лицом.")
    print(f"Кадры: {frames_dir}")
    print(f"Описания: {json_path}")
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Построение датасета с 478 лицевыми точками")
    parser.add_argument("video", help="Путь к видеофайлу (например, input.mp4)")
    parser.add_argument(
        "--out",
        default="output_dataset",
        help="Папка для сохранения датасета (по умолчанию output_dataset)",
    )

    args = parser.parse_args()
    build_dataset(args.video, args.out)
