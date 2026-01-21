import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

DATASET_DIR = "dataset"
OUTPUT_DIR = "embeddings"

print("Dang khoi tao model (AMD DirectML)...")
app = FaceAnalysis(name='buffalo_l', providers=['DmlExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
handler = app.models['recognition']

def create_embedding(person_name, person_folder):
    save_path = os.path.join(OUTPUT_DIR, f"{person_name}.npy")
    if os.path.exists(save_path):
        print(f"  [>>>] BO QUA: {person_name} (File .npy da ton tai)")
        return

    print(f"\nDang vector hoa NEW: {person_name}...")
    
    embeddings = []
    image_files = [f for f in os.listdir(person_folder) if f.endswith(('.jpg', '.png'))]
    
    for img_file in image_files:
        img_path = os.path.join(person_folder, img_file)
        img = cv2.imread(img_path)
        if img is None: continue

        try:
            input_face = cv2.resize(img, (112, 112))
            embedding = handler.get_feat(input_face)
            if embedding is not None:
                embeddings.append(embedding)
        except Exception:
            pass

    if len(embeddings) == 0:
        print("  [Loi] Khong rut trich duoc vector!")
        return

    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

    np.save(save_path, mean_embedding)
    print(f"  [OK] Da tao moi: {save_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    if not os.path.exists(DATASET_DIR):
        print("Khong tim thay folder dataset!")
        return

    sub_folders = [f.path for f in os.scandir(DATASET_DIR) if f.is_dir()]
    
    print(f"Quet thay {len(sub_folders)} doi tuong trong dataset.")
    
    for folder in sub_folders:
        create_embedding(os.path.basename(folder), folder)

    print("\n=== HOAN TAT TAO EMBEDDINGS ===")

if __name__ == "__main__":
    main()