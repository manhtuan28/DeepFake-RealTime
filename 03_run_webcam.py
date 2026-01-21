import cv2
import insightface
import numpy as np
import os
import time

EMBEDDINGS_DIR = "embeddings"
MODEL_PATH = "inswapper_128.onnx"
CONFIDENCE_THRESHOLD = 0.5

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

if not os.path.exists(MODEL_PATH):
    print("ERROR: Khong tim thay file model inswapper_128.onnx")
    exit()

print(">>> Dang khoi tao he thong Split View...")

face_bank = {}
files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith('.npy')]
if not files:
    print(f"ERROR: Khong tim thay file .npy trong {EMBEDDINGS_DIR}!")
    exit()

for idx, f in enumerate(files):
    name = os.path.splitext(f)[0]
    face_bank[idx] = {
        "name": name,
        "embedding": np.load(os.path.join(EMBEDDINGS_DIR, f))
    }
    print(f"  [Loaded]: {name}")

current_face_idx = 0

app = insightface.app.FaceAnalysis(name='buffalo_s', providers=['DmlExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(MODEL_PATH, providers=['DmlExecutionProvider'])

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("\n>>> HE THONG SAN SANG! <<<")
print("Bam 'N' de doi nguoi. Bam 'Q' de thoat.")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)

    faces = app.get(frame)
    current_name = face_bank[current_face_idx]['name']
    source_face_wrapper = insightface.app.common.Face(embedding=face_bank[current_face_idx]['embedding'])

    fake_frame = frame.copy()

    for face in faces:
        if face.det_score > CONFIDENCE_THRESHOLD:
            fake_frame = swapper.get(fake_frame, face, source_face_wrapper, paste_back=True)
    
    fake_frame = sharpen_image(fake_frame)

    combined_window = np.hstack((frame, fake_frame))

    cv2.line(combined_window, (FRAME_WIDTH, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 255, 0), 2)

    cv2.putText(combined_window, "ORIGINA", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(combined_window, f"FAKE: {current_name.upper()}", (FRAME_WIDTH + 30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    fps = 1 / (time.time() - start_time)
    info_text = f"FPS: {fps:.1f}"
    cv2.putText(combined_window, info_text, (FRAME_WIDTH - 200, FRAME_HEIGHT - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow('Deep Fake: Real-Time', combined_window)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        current_face_idx = (current_face_idx + 1) % len(face_bank)
        print(f"-> Doi: {face_bank[current_face_idx]['name']}")

cap.release()
cv2.destroyAllWindows()