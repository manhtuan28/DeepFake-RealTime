import cv2
import os
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align

INPUT_VIDEO_DIR = "video_data"
OUTPUT_DATASET_DIR = "dataset"
FRAME_INTERVAL = 10
BLUR_THRESHOLD = 80
MIN_FACE_SIZE = 100
OUTPUT_SIZE = 512
SIMILARITY_THRESHOLD = 0.5

print(">>> Dang khoi tao AI Smart Filter (CPU Mode)...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def compute_sim(feat1, feat2):
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

def check_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_target_embedding(image_path):
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path)
    if img is None: return None
    
    faces = app.get(img)
    if len(faces) == 0:
        print("ERROR: Khong tim thay mat trong anh mau!")
        return None
    
    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    return faces[0].normed_embedding

def process_video(video_path, target_embedding=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_folder = os.path.join(OUTPUT_DATASET_DIR, video_name)
    
    if os.path.exists(save_folder) and len(os.listdir(save_folder)) > 0:
        print(f"[SKIP] Da co data: {video_name}")
        return

    os.makedirs(save_folder, exist_ok=True)
    print(f"\n--- Dang xu ly: {video_name} ---")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        count += 1
        if count % FRAME_INTERVAL != 0: continue

        if check_blur(frame) < BLUR_THRESHOLD: continue

        try:
            faces = app.get(frame)
        except: continue

        if len(faces) == 0: continue

        for face in faces:
            bbox = face.bbox
            if (bbox[2] - bbox[0]) < MIN_FACE_SIZE: continue

            if target_embedding is not None:
                sim = compute_sim(face.normed_embedding, target_embedding)
                if sim < SIMILARITY_THRESHOLD:
                    continue

            try:
                norm_crop_img = face_align.norm_crop(frame, landmark=face.kps, image_size=OUTPUT_SIZE)
                file_name = f"{video_name}_{saved_count:04d}.jpg"
                cv2.imwrite(os.path.join(save_folder, file_name), norm_crop_img)
                saved_count += 1
                print(f"\rDa luu: {saved_count} | Frame: {count}/{total_frames}", end="")
            except:
                pass

    cap.release()
    print(f"\n[DONE] Video {video_name}: {saved_count} anh.")

def main():
    if not os.path.exists(INPUT_VIDEO_DIR):
        os.makedirs(INPUT_VIDEO_DIR)
        return

    print("\n[CHẾ ĐỘ LỌC NGƯỜI]")
    use_filter = input("Cậu có muốn lọc người cụ thể không? (y/n): ").strip().lower()
    
    target_emb = None
    if use_filter == 'y':
        print("Hãy chuẩn bị 1 tấm ảnh mẫu (JPG/PNG) của người đó.")
        img_path = input("Nhập tên file ảnh mẫu (ví dụ: mau.jpg): ").strip()
        
        if not os.path.exists(img_path):
            img_path = os.path.join(INPUT_VIDEO_DIR, img_path)
        
        print(f"Dang doc vector khuon mat mau tu: {img_path}...")
        target_emb = get_target_embedding(img_path)
        
        if target_emb is None:
            print("LỖI: Không đọc được ảnh mẫu. Sẽ cắt TOÀN BỘ người trong video.")
        else:
            print("OK! Đã học xong mặt mẫu. Bắt đầu lọc...")

    video_files = [f for f in os.listdir(INPUT_VIDEO_DIR) if f.endswith(('.mp4', '.avi'))]
    
    for vid in video_files:
        process_video(os.path.join(INPUT_VIDEO_DIR, vid), target_emb)

if __name__ == "__main__":
    main()