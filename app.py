import sys
import os
import subprocess
import time


TEXTS = {
    "vi": {
        "title": "HỆ THỐNG SWAP MẶT - TRUNG TÂM ĐIỀU KHIỂN",
        "subtitle": "(Auto backend: CUDA / DirectML / CPU)",
        "pick_lang": "Chọn ngôn ngữ / Choose language:",
        "lang_vi": "1. Tiếng Việt",
        "lang_en": "2. English",
        "lang_prompt": "Nhập lựa chọn (1-2, mặc định 1): ",
        "back_enter": "Nhấn Enter để quay lại menu...",
        "features": "CHỌN CHỨC NĂNG:",
        "menu_1": "1. [DATASET]  Xử lý ảnh thô (Raw -> Dataset)",
        "menu_1_sub": "   (Dùng khi mới thêm ảnh vào folder raw_data)",
        "menu_2": "2. [VECTOR]   Tạo dữ liệu AI (Dataset -> Embeddings)",
        "menu_2_sub": "   (Dùng sau khi bước 1 chạy xong)",
        "menu_3": "3. [START]    Chạy Face Swap (Webcam Real-time)",
        "menu_3_sub": "   (Chế độ màn hình đôi: Thật vs Giả)",
        "menu_4": "4. [VIDEO]    Trích xuất ảnh từ video",
        "menu_4_sub": "   (Nhảy file .mp4 vào folder video_data -> Tự tạo Dataset)",
        "menu_5": "5. [RENDER]   Swap mặt vào video có sẵn",
        "menu_5_sub": "   (Đặt file video input trong thư mục hiện tại)",
        "menu_6": "6. [HEAD]     Fake toàn bộ đầu (LivePortrait)",
        "menu_6_sub": "   (Chậm hơn, nhưng phủ cả đầu thay vì chỉ mặt)",
        "menu_7": "7. [MODEL]    Tự động tải model theo cấu hình máy",
        "menu_7_sub": "   (Máy yếu: gọi nhẹ, máy mạnh: gọi đầy đủ)",
        "menu_0": "0. Thoát",
        "choice": "Nhập lựa chọn (0-7): ",
        "starting": ">>> Đang khởi động: {script}...",
        "file_missing": "[LỖI] Không tìm thấy file: {script}",
        "script_error": "[LỖI] Có lỗi khi chạy script: {err}",
        "script_stopped": "[THÔNG TIN] Đã dừng script.",
        "done_enter": ">>> Đã xong! Nhấn Enter để quay lại menu chính...",
        "invalid": "Lựa chọn không hợp lệ!",
        "exit": "Đã thoát.",
    },
    "en": {
        "title": "AI FACE SWAP SYSTEM - CONTROL CENTER",
        "subtitle": "(Auto backend: CUDA / DirectML / CPU)",
        "pick_lang": "Choose language / Chọn ngôn ngữ:",
        "lang_vi": "1. Tiếng Việt",
        "lang_en": "2. English",
        "lang_prompt": "Select option (1-2, default 2): ",
        "back_enter": "Press Enter to return to menu...",
        "features": "SELECT FEATURES:",
        "menu_1": "1. [DATASET]  Process raw images (Raw -> Dataset)",
        "menu_1_sub": "   (Use this after adding images into raw_data)",
        "menu_2": "2. [VECTOR]   Build AI vectors (Dataset -> Embeddings)",
        "menu_2_sub": "   (Run after step 1)",
        "menu_3": "3. [START]    Run Face Swap (Webcam Real-time)",
        "menu_3_sub": "   (Split view mode: Real vs Fake)",
        "menu_4": "4. [VIDEO]    Extract faces from video",
        "menu_4_sub": "   (Drop video files into video_data to build dataset)",
        "menu_5": "5. [RENDER]   Swap face into existing video",
        "menu_5_sub": "   (Use video input from current folder)",
        "menu_6": "6. [HEAD]     Whole-head mode (LivePortrait)",
        "menu_6_sub": "   (Slower but covers full head instead of face only)",
        "menu_7": "7. [MODEL]    Auto setup models by machine profile",
        "menu_7_sub": "   (Weak machine: lightweight set, strong machine: full set)",
        "menu_0": "0. Exit",
        "choice": "Enter your choice (0-7): ",
        "starting": ">>> Starting: {script}...",
        "file_missing": "[ERROR] File not found: {script}",
        "script_error": "[ERROR] Script failed: {err}",
        "script_stopped": "[INFO] Script stopped.",
        "done_enter": ">>> Done! Press Enter to return to the main menu...",
        "invalid": "Invalid option!",
        "exit": "Exited.",
    },
}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def choose_language() -> str:
    clear_screen()
    print("=" * 50)
    print(TEXTS["en"]["pick_lang"])
    print(TEXTS["en"]["lang_vi"])
    print(TEXTS["en"]["lang_en"])
    print("=" * 50)
    choice = input(TEXTS["en"]["lang_prompt"]).strip()
    if choice == "1":
        return "vi"
    return "en"


def t(lang: str, key: str) -> str:
    return TEXTS.get(lang, TEXTS["en"]).get(key, key)


def print_header(lang: str):
    print("\033[92m" + "="*50)
    print(f"      {t(lang, 'title')}")
    print(f"      {t(lang, 'subtitle')}")
    print("="*50 + "\033[0m")


def run_script(script_name: str, lang: str):
    if not os.path.exists(script_name):
        print(f"\n\033[91m{t(lang, 'file_missing').format(script=script_name)}\033[0m")
        input(t(lang, "back_enter"))
        return

    print(f"\n{t(lang, 'starting').format(script=script_name)}\n")
    child_env = os.environ.copy()
    child_env["DEEPFAKE_LANG"] = lang
    
    try:
        subprocess.run([sys.executable, script_name], check=True, env=child_env)
    except subprocess.CalledProcessError as e:
        print(f"\n{t(lang, 'script_error').format(err=e)}")
    except KeyboardInterrupt:
        print(f"\n{t(lang, 'script_stopped')}")
    
    print("\n" + "-"*30)
    input(t(lang, "done_enter"))

def main():
    lang = choose_language()
    while True:
        clear_screen()
        print_header(lang)
        
        print(f"\n{t(lang, 'features')}")
        print(f"\033[96m{t(lang, 'menu_1')}\033[0m")
        print(t(lang, "menu_1_sub"))
        
        print(f"\n\033[96m{t(lang, 'menu_2')}\033[0m")
        print(t(lang, "menu_2_sub"))
        
        print(f"\n\033[93m{t(lang, 'menu_3')}\033[0m")
        print(t(lang, "menu_3_sub"))

        print("-" * 30)
        
        print(f"\n\033[95m{t(lang, 'menu_4')}\033[0m")
        print(t(lang, "menu_4_sub"))

        print(f"\n\033[91m{t(lang, 'menu_5')}\033[0m")
        print(t(lang, "menu_5_sub"))

        print(f"\n\033[94m{t(lang, 'menu_6')}\033[0m")
        print(t(lang, "menu_6_sub"))

        print(f"\n\033[92m{t(lang, 'menu_7')}\033[0m")
        print(t(lang, "menu_7_sub"))
        
        print(f"\n\033[90m{t(lang, 'menu_0')}\033[0m")
        
        print("-" * 50)
        choice = input(t(lang, "choice")).strip()

        if choice == '1':
            run_script('01_data_processing.py', lang)
        elif choice == '2':
            run_script('02_create_embeddings.py', lang)
        elif choice == '3':
            run_script('03_run_webcam.py', lang)
        elif choice == '4':
            run_script('04_video_to_dataset.py', lang)
        elif choice == '5':
            run_script('05_run_video_file.py', lang)
        elif choice == '6':
            run_script('07_head_stitcher.py', lang)
        elif choice == '7':
            run_script('06_setup_models.py', lang)
        elif choice == '0':
            print(f"\n{t(lang, 'exit')}")
            break
        else:
            print(f"\n{t(lang, 'invalid')}")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã thoát chương trình.")