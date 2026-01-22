import cv2
import insightface
import numpy as np
import os
import sys

EMBEDDINGS_DIR = "embeddings"
MODEL_SWAP_PATH = "inswapper_128.onnx"
MODEL_ENHANCE_PATH = "GPEN-BFR-512.onnx"

def select_file_from_list(file_list, prompt_text):
    if not file_list:
        print(f"\n[LỖI] Không tìm thấy file nào ({prompt_text})!")
        return None

    print(f"\n--- CHỌN {prompt_text.upper()} ---")
    for i, f in enumerate(file_list):
        print(f"{i + 1}. {f}")
    
    while True:
        try:
            choice = int(input(f"\n>> Nhập số thứ tự (1-{len(file_list)}): "))
            if 1 <= choice <= len(file_list):
                return file_list[choice - 1]
            else:
                print("Số không hợp lệ, vui lòng chọn lại.")
        except ValueError:
            print("Vui lòng nhập số.")

def get_video_files():
    extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
    return [f for f in os.listdir('.') if f.lower().endswith(extensions) and not f.startswith("output_")]

def get_embedding_files():
    if not os.path.exists(EMBEDDINGS_DIR):
        return []
    return [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith('.npy')]

def main():
    print("="*50)
    print("   VIDEO FACE SWAP - LITE VERSION (NO ENHANCE)")
    print("="*50)

    if not os.path.exists(MODEL_SWAP_PATH):
        print(f"\n[LỖI] Không tìm thấy file model: {MODEL_SWAP_PATH}")
        return

    video_files = get_video_files()
    if not video_files:
        print("[LỖI] Không tìm thấy video nào trong thư mục này.")
        return
    input_video = select_file_from_list(video_files, "Video nguồn")
    print(f"-> Đã chọn video: {input_video}")

    emb_files = get_embedding_files()
    if not emb_files:
        print(f"[LỖI] Thư mục '{EMBEDDINGS_DIR}' trống trơn.")
        return
    target_file = select_file_from_list(emb_files, "Khuôn mặt muốn Fake")
    target_name = os.path.splitext(target_file)[0]
    print(f"-> Đã chọn mặt: {target_name}")

    output_file = f"output_lite_{input_video}"
    target_embedding = np.load(os.path.join(EMBEDDINGS_DIR, target_file))
    source_face = insightface.app.common.Face(embedding=target_embedding)

    print("\n>>> Đang khởi động AI Models (Classic)...")
    providers = ['DmlExecutionProvider']
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(MODEL_SWAP_PATH, providers=providers)
    
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"\n>>> BẮT ĐẦU RENDER: {width}x{height} @ {fps}fps")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        try:
            faces = app.get(frame)
            res_frame = frame.copy()

            for face in faces:
                res_frame = swapper.get(res_frame, face, source_face, paste_back=True)

            out.write(res_frame)
            
            percent = (frame_count / total_frames) * 100
            bar = '#' * int(percent / 5) + '-' * (20 - int(percent / 5))
            print(f"\r[{bar}] {percent:.1f}% | Frame: {frame_count}/{total_frames}", end="")
            
        except Exception:
            out.write(frame)

    cap.release()
    out.release()
    print(f"\n\n[XONG] Video đã lưu tại: {output_file}")
    input("\nẤn Enter để thoát...")

if __name__ == "__main__":
    main()