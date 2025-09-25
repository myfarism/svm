from flask import Flask, render_template, Response, jsonify
import cv2
import dlib
import numpy as np
import pickle
import threading

app = Flask(__name__)

# Load model deteksi wajah dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load model SVM dan Label Encoder
try:
    with open("svm_model_fix.pkl", "rb") as f:
        svm = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

try:
    with open("label_fix.pkl", "rb") as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading models label: {e}")
    exit()


# Inisialisasi kamera RTSP dengan FFMPEG
rtsp_url = "rtsp://admin:admin@172.13.14.164:1935"
cap = cv2.VideoCapture(0)

# Kurangi buffer untuk menghindari delay panjang
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# Set resolusi jika diperlukan
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detected_names = set()  # Gunakan set untuk menyimpan nama unik
lock = threading.Lock()  # Menghindari race condition saat update data

def generate_frames():
    global detected_names

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_frame)

        with lock:  # Menggunakan lock agar thread-safe
            detected_names.clear()  

            for face in faces:
                shape = sp(rgb_frame, face)
                face_descriptor = facerec.compute_face_descriptor(rgb_frame, shape)
                face_descriptor = np.array(face_descriptor).reshape(1, -1)

                pred_label = svm.predict(face_descriptor)[0]
                pred_name = label_encoder.inverse_transform([pred_label])[0]

                detected_names.add(pred_name)

                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, pred_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_faces')
def get_detected_faces():
    with lock:
        return jsonify(list(detected_names))

@app.route('/clear_faces', methods=['POST'])
def clear_faces():
    with lock:
        detected_names.clear()
    return jsonify({"message": "Detected faces cleared."})

if __name__ == "__main__":
    app.run(debug=True)
