import sys
import os
import subprocess
import time

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("\033[92m" + "="*50)
    print("      AI FACE SWAP SYSTEM - CONTROL CENTER")
    print("      (AMD DirectML Optimized Edition)")
    print("="*50 + "\033[0m")

def run_script(script_name):
    if not os.path.exists(script_name):
        print(f"\n\033[91m[LỖI] Không tìm thấy file: {script_name}\033[0m")
        input("Nhấn Enter để quay lại menu...")
        return

    print(f"\n>>> Đang khởi động: {script_name}...\n")
    
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[LỖI] Có lỗi xảy ra khi chạy script: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Đã dừng script.")
    
    print("\n" + "-"*30)
    input(">>> Đã xong! Nhấn Enter để quay lại menu chính...")

def main():
    while True:
        clear_screen()
        print_header()
        
        print("\nCHỌN CHỨC NĂNG:")
        print("1. \033[96m[DATASET]\033[0m  Xử lý ảnh thô (Raw -> Dataset)")
        print("   (Dùng khi mới thêm ảnh vào folder raw_data)")
        
        print("\n2. \033[96m[VECTOR]\033[0m   Tạo dữ liệu AI (Dataset -> Embeddings)")
        print("   (Dùng sau khi bước 1 chạy xong)")
        
        print("\n3. \033[93m[START]\033[0m    CHẠY FACE SWAP (Webcam Real-time)")
        print("   (Chế độ màn hình đôi: Thật vs Fake)")

        print("-" * 30)
        
        print("4. \033[95m[VIDEO]\033[0m   Trích xuất ảnh từ Video (Youtube Interview)")
        print("   (Ném file .mp4 vào folder video_data -> Tự tạo Dataset)")
        
        print("\n0. \033[90mThoát\033[0m")
        
        print("-" * 50)
        choice = input("Nhập lựa chọn (0-3): ").strip()

        if choice == '1':
            run_script('01_data_processing.py')
        elif choice == '2':
            run_script('02_create_embeddings.py')
        elif choice == '3':
            run_script('03_run_webcam.py')
        elif choice == '4':
            run_script('04_video_to_dataset.py')
        elif choice == '0':
            print("\nCút!")
            break
        else:
            print("\nLựa chọn không hợp lệ!")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã thoát chương trình.")