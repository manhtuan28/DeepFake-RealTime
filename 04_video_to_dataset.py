import cv2
import os
import numpy as np
from insightface.utils import face_align
from runtime_utils import create_face_analysis

INPUT_VIDEO_DIR = "video_data"
OUTPUT_DATASET_DIR = "dataset"
FRAME_INTERVAL = 5
BLUR_THRESHOLD = 20
MIN_FACE_SIZE = 60
OUTPUT_SIZE = 512
SIMILARITY_THRESHOLD = 0.3

print(">>> Dang khoi tao AI Smart Filter (Optimized for High-Res)...")
app = create_face_analysis(model_name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

def compute_sim(feat1, feat2):
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

def check_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_target_embedding(image_path):
    if not os.path.exists(image_path): return None
    img = cv2.imread(image_path)
    if img is None: return None
    faces = app.get(img)
    if len(faces) == 0: return None
    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    return faces[0].normed_embedding

def process_video(video_path, target_embedding=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_folder = os.path.join(OUTPUT_DATASET_DIR, video_name)
    if os.path.exists(save_folder) and len(os.listdir(save_folder)) > 0:
        print(f"[SKIP] Da co data: {video_name}")
        return

    os.makedirs(save_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        count += 1
        if count % FRAME_INTERVAL != 0: continue

        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280.0 / w
            frame_small = cv2.resize(frame, (1280, int(h * scale)))
        else:
            frame_small = frame.copy()

        if check_blur(frame_small) < BLUR_THRESHOLD: continue

        try:
            faces = app.get(frame_small)
        except: continue

        for face in faces:
            if w > 1280:
                scale_inv = w / 1280.0
                face.bbox = face.bbox * scale_inv
                face.kps = face.kps * scale_inv

            bbox = face.bbox
            if (bbox[2] - bbox[0]) < MIN_FACE_SIZE: continue

            if target_embedding is not None:
                sim = compute_sim(face.normed_embedding, target_embedding)
                if sim < SIMILARITY_THRESHOLD: continue

            try:
                norm_crop_img = face_align.norm_crop(frame, landmark=face.kps, image_size=OUTPUT_SIZE)
                file_name = f"{video_name}_{saved_count:04d}.jpg"
                cv2.imwrite(os.path.join(save_folder, file_name), norm_crop_img)
                saved_count += 1
                print(f"\rRender: {saved_count} | Frame: {count}/{total_frames} ({int(count/total_frames*100)}%)", end="")
            except: pass

    cap.release()
    print(f"\n[DONE] {video_name}: {saved_count} anh.")

def main():
    if not os.path.exists(INPUT_VIDEO_DIR): os.makedirs(INPUT_VIDEO_DIR)
    
    print("\n[CHẾ ĐỘ LỌC NGƯỜI TỐI ƯU 4K/8K]")
    use_filter = input("Cậu có muốn lọc người cụ thể không? (y/n): ").strip().lower()
    
    target_emb = None
    if use_filter == 'y':
        img_path = input("Nhập tên file ảnh mẫu (VD: mau.jpg): ").strip()
        if not os.path.exists(img_path):
            img_path = os.path.join(INPUT_VIDEO_DIR, img_path)
        target_emb = get_target_embedding(img_path)
        if target_emb is None:
            print("LỖI: Không tìm thấy mặt mẫu. Sẽ cắt TOÀN BỘ.")
        else:
            print("OK! Đã học xong mặt mẫu.")

    video_files = [f for f in os.listdir(INPUT_VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    for vid in video_files:
        process_video(os.path.join(INPUT_VIDEO_DIR, vid), target_emb)

if __name__ == "__main__":
    main()