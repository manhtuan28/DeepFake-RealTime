import os
import cv2
import glob
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align

INPUT_DIR = "raw_data"
OUTPUT_DIR = "dataset"
MIN_FACE_SIZE = 100
OUTPUT_SIZE = 512

app = FaceAnalysis(allowed_modules=['detection'], providers=['CPUExecutionProvider']) 
app.prepare(ctx_id=0, det_size=(640, 640))

def process_image(img_path, save_folder, count_start):
    img = cv2.imread(img_path)
    if img is None: return count_start

    try:
        faces = app.get(img)
    except Exception:
        return count_start

    if len(faces) == 0: return count_start

    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    
    if len(faces) > 0:
        face = faces[0]
        bbox = face.bbox
        if (bbox[2] - bbox[0]) < MIN_FACE_SIZE:
            return count_start

        norm_crop_img = face_align.norm_crop(img, landmark=face.kps, image_size=OUTPUT_SIZE)

        file_name = f"{os.path.basename(save_folder)}_{count_start:04d}.jpg"
        save_path = os.path.join(save_folder, file_name)
        cv2.imwrite(save_path, norm_crop_img)
        
        print(f"    + Saved: {file_name}")
        return count_start + 1

    return count_start

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Chua co thu muc {INPUT_DIR}!")
        return

    sub_folders = [f.path for f in os.scandir(INPUT_DIR) if f.is_dir()]

    for folder in sub_folders:
        person_name = os.path.basename(folder)
        target_folder = os.path.join(OUTPUT_DIR, person_name)

        if os.path.exists(target_folder) and len(os.listdir(target_folder)) > 0:
            print(f"\n[SKIP] {person_name} (Da co data)")
            continue

        print(f"\n--- Dang xu ly: {person_name} ---")
        os.makedirs(target_folder, exist_ok=True)

        types = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
        files_grabbed = []
        for files in types:
            files_grabbed.extend(glob.glob(os.path.join(folder, files)))

        if len(files_grabbed) == 0:
            print("   ! Khong tim thay anh nao (Check lai duoi file)")
            continue

        count = 1
        for img_file in files_grabbed:
            count = process_image(img_file, target_folder, count)

    print("\n=== HOAN TAT XU LY ===")

if __name__ == "__main__":
    main()