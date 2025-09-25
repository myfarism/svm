import cv2
import os

# Path ke folder dataset
dataset_path = "dataset"

# Loop semua file di dalam folder dataset
for filename in os.listdir(dataset_path):
    if filename.endswith(".mp4"):
        video_path = os.path.join(dataset_path, filename)
        video_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(dataset_path, video_name)

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()
        print(f"Berhasil ekstrak {frame_count} frame dari {filename} ke {output_dir}")
