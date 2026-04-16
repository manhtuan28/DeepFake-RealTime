import cv2
import insightface
import numpy as np
import os
import time
from runtime_utils import create_face_analysis, get_onnxruntime_providers, open_video_capture

EMBEDDINGS_DIR = "embeddings"
MODEL_PATH = os.path.join("models", "inswapper_128.onnx")
CONFIDENCE_THRESHOLD = 0.5

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

COLOR_CYAN = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BG_TXT = (0, 0, 0)

if not os.path.exists(MODEL_PATH):
    print("ERROR: Khong tim thay file model models/inswapper_128.onnx")
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

providers = get_onnxruntime_providers()
print(f">>> Selected providers: {', '.join(providers)}")

app = create_face_analysis(model_name='buffalo_s')
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(MODEL_PATH, providers=providers)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def draw_ui_text(img, text, pos, color, scale=0.7, thickness=2):
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 5, y - h - 10), (x + w + 5, y + 5), COLOR_BG_TXT, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

cap = open_video_capture(0)
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

    cv2.line(combined_window, (FRAME_WIDTH, 0), (FRAME_WIDTH, FRAME_HEIGHT), (200, 200, 200), 2)

    draw_ui_text(combined_window, "REAL CAMERA", (20, 40), COLOR_WHITE)

    fps = 1 / (time.time() - start_time)
    
    overlay = combined_window.copy()
    cv2.rectangle(overlay, (0, FRAME_HEIGHT - 40), (FRAME_WIDTH * 2, FRAME_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, combined_window, 0.4, 0, combined_window)

    cv2.putText(combined_window, f"FPS: {fps:.1f}", (20, FRAME_HEIGHT - 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CYAN, 1)

    help_text = "[N]: Next Person  |  [Q]: Quit"
    (w_help, _), _ = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(combined_window, help_text, ((FRAME_WIDTH * 2) - w_help - 20, FRAME_HEIGHT - 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

    cv2.imshow('Deep Fake: Real-Time UI', combined_window)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        current_face_idx = (current_face_idx + 1) % len(face_bank)
        print(f"-> Doi: {face_bank[current_face_idx]['name']}")

cap.release()
cv2.destroyAllWindows()