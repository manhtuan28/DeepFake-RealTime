import cv2
import numpy as np
import os
from runtime_utils import create_onnx_session, get_onnxruntime_providers, open_video_capture

# --- CẤU HÌNH ---
EMBEDDINGS_DIR = "embeddings"
RAW_DATA_DIR = "raw_data"
MODEL_DIR = "models/liveportrait"
PROVIDERS = get_onnxruntime_providers()
SOURCE_SIZE = int(os.getenv("DEEPFAKE_HEAD_SOURCE_SIZE", "224"))
DRIVING_SIZE = int(os.getenv("DEEPFAKE_HEAD_DRIVING_SIZE", "192"))
FRAME_SKIP = max(1, int(os.getenv("DEEPFAKE_HEAD_FRAME_SKIP", "2")))

class HeadStitcher:
    def __init__(self):
        print(f">>> Đang khởi tạo bộ não LivePortrait ({', '.join(PROVIDERS)})...")
        # Khởi tạo các session ONNX
        self.appearance_feat = create_onnx_session(f"{MODEL_DIR}/appearance_feature_extractor.onnx")
        self.motion_ext = create_onnx_session(f"{MODEL_DIR}/motion_extractor.onnx")
        self.warping_net = create_onnx_session(f"{MODEL_DIR}/warping_spatially_adaptive_network.onnx")
        self.stitcher = create_onnx_session(f"{MODEL_DIR}/stitching_retargeting.onnx")

    def get_source_image(self, person_name):
        person_path = os.path.join(RAW_DATA_DIR, person_name)
        if not os.path.exists(person_path): return None
        valid_ext = ('.jpg', '.jpeg', '.png', '.webp')
        for f in os.listdir(person_path):
            if f.lower().endswith(valid_ext):
                return cv2.imread(os.path.join(person_path, f))
        return None

    def preprocess(self, img, size=256):
        img = cv2.resize(img, (size, size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def run(self):
        # 1. Quét Embeddings
        files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith('.npy')]
        if not files:
            print(f"LỖI: Không tìm thấy dữ liệu trong {EMBEDDINGS_DIR}!")
            return

        print("\n--- DANH SÁCH NHÂN VẬT ---")
        for i, f in enumerate(files):
            print(f"{i + 1}. {os.path.splitext(f)[0].upper()}")

        try:
            idx = int(input(f"\n>> Chọn số (1-{len(files)}): ")) - 1
            person_name = os.path.splitext(files[idx])[0]
        except: return

        # 2. Chuẩn bị dữ liệu Idol
        source_img = self.get_source_image(person_name)
        if source_img is None:
            print(f"LỖI: Thiếu ảnh gốc cho {person_name}!")
            return

        source_tensor = self.preprocess(source_img)
        feature_3d = self.appearance_feat.run(None, {'img': source_tensor})[0]
        kp_source = self.motion_ext.run(None, {'img': source_tensor})[0]

        # 3. Mở Webcam
        cap = open_video_capture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"\n>>> ĐANG FAKE CẢ ĐẦU THÀNH: {person_name.upper()}")

        frame_index = 0
        cached_output = None

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            frame_index += 1
            if frame_index % FRAME_SKIP != 0 and cached_output is not None:
                cv2.imshow("Head Swap Live - Whole Head Mode", cached_output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            d_tensor = self.preprocess(frame, size=DRIVING_SIZE)
            kp_driving = self.motion_ext.run(None, {'img': d_tensor})[0]

            try:
                # --- 1. CHUẨN BỊ INPUT 126 ---
                kps_flat = kp_source.flatten()[:63] # Lấy 63 giá trị đầu (21 pts * 3)
                kpd_flat = kp_driving.flatten()[:63]
                
                # Ghép lại thành mảng 126
                combined = np.concatenate([kps_flat, kpd_flat]).astype(np.float32)
                if combined.shape[0] < 126:
                    combined = np.pad(combined, (0, 126 - combined.shape[0]))
                
                # Đưa về Rank 2: [1, 126]
                input_stitcher = combined.reshape(1, 126)
                
                # --- 2. CHẠY STITCHER (Kết quả ra (1, 65)) ---
                kp_combined_raw = self.stitcher.run(None, {'input': input_stitcher})[0]

                # --- 3. XỬ LÝ TRỤC (AXES) TỪ 65 VỀ 63 ---
                # Chuyển từ (1, 65) -> (65,) -> (63,)
                kp_feat = kp_combined_raw.ravel() # Đưa về 1 chiều phẳng lặng
                kp_feat_63 = kp_feat[:63]         # Cắt lấy đúng 21 điểm * 3 trục
                
                # Ép về Rank 3 chuẩn: [1, 21, 3] cho Warping Net
                kp_combined_v3 = kp_feat_63.reshape(1, 21, 3).astype(np.float32)
                
                # Tương tự cho kp_source (Ảnh Idol)
                kps_feat_63 = kp_source.ravel()[:63]
                if kps_feat_63.shape[0] < 63:
                    kps_feat_63 = np.pad(kps_feat_63, (0, 63 - kps_feat_63.shape[0]))
                kps_v3 = kps_feat_63.reshape(1, 21, 3).astype(np.float32)

                # --- 4. RENDER VỚI WARPING NET ---
                output_tensor = self.warping_net.run(None, {
                    'feature_3d': feature_3d.astype(np.float32), 
                    'kp_source': kps_v3,
                    'kp_driving': kp_combined_v3
                })[0]

                # Hiển thị
                out = np.squeeze(output_tensor).transpose(1, 2, 0)
                out = (out * 255.0).clip(0, 255).astype(np.uint8)
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

                cached_output = out
                cv2.imshow("Head Swap Live - Whole Head Mode", out)
                
            except Exception as e:
                print(f"\rLỗi xử lý: {e} | Shape raw: {kp_combined_raw.shape if 'kp_combined_raw' in locals() else 'N/A'}", end="")

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    stitcher = HeadStitcher()
    stitcher.run()