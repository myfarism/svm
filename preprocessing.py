import cv2
import os
from tqdm import tqdm  

# Path folder frame hasil ekstraksi
frames_folder = "dataset"
processed_folder = "processed"
os.makedirs(processed_folder, exist_ok=True)

# Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Loop dengan tqdm untuk menampilkan progress
for root, dirs, files in os.walk(frames_folder):
    for file in tqdm(files, desc="Processing frames", unit="frame"):
        if file.endswith((".jpg", ".png")):
            # Menentukan path gambar
            img_path = os.path.join(root, file)
            
            # Mengambil nama folder untuk subfolder (misalnya 'nim' dari 'data-tambahan/nim')
            nim_folder = os.path.basename(root)
            
            # Membuat subfolder untuk nim di folder processed_faces_fix
            nim_processed_folder = os.path.join(processed_folder, nim_folder)
            os.makedirs(nim_processed_folder, exist_ok=True)
            
            # Membaca gambar dan konversi ke grayscale
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Deteksi wajah
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (160, 160))
                
                # Tentukan path untuk menyimpan hasil wajah
                save_path = os.path.join(nim_processed_folder, file)
                cv2.imwrite(save_path, face)

print("Proses selesai!")
