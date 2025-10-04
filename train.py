import os
import cv2
import dlib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Path dataset
DATASET_PATH = "preprocessed"

# List untuk menyimpan fitur wajah dan label
face_descriptors = []
labels = []

# Loop setiap folder dalam dataset dengan tqdm untuk progress bar
for person in tqdm(os.listdir(DATASET_PATH), desc="Processing dataset folders"):
    person_path = os.path.join(DATASET_PATH, person)
    
    if os.path.isdir(person_path):
        for image_name in tqdm(os.listdir(person_path), desc=f"Processing images for {person}", leave=False):
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Deteksi wajah
            faces = detector(rgb_img)
            for face in faces:
                shape = sp(rgb_img, face)
                face_descriptor = facerec.compute_face_descriptor(rgb_img, shape)
                face_descriptors.append(np.array(face_descriptor))
                labels.append(person)

# Konversi ke NumPy array
X = np.array(face_descriptors)
y = np.array(labels)

# Encode label menjadi angka
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Simpan encoder label
with open("label_fix.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Train model SVM
svm = SVC(kernel="linear", probability=True)
svm.fit(X, y_encoded)

# Simpan fitur wajah dan label ke file
with open("dataset_features_fix.pkl", "wb") as f:
    pickle.dump({"features": X, "labels": y_encoded}, f)

# Simpan model SVM dengan pickle
with open("svm_model_fix.pkl", "wb") as f:
    pickle.dump(svm, f)

print("Model berhasil dilatih dan disimpan!")

# Convert model SVM ke format ONNX
# Tentukan input type untuk model
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(svm, initial_types=initial_type)

# Simpan model SVM dalam format ONNX
onnx.save_model(onnx_model, "svm_model_fix.onnx")
print("Model SVM berhasil disimpan dalam format ONNX!")
