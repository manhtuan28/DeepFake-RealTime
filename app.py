import sys
import os
import subprocess
import time


TEXTS = {
    "vi": {
        "title": "AI FACE SWAP SYSTEM - CONTROL CENTER",
        "subtitle": "(Auto backend: CUDA / DirectML / CPU)",
        "pick_lang": "Chon ngon ngu / Choose language:",
        "lang_vi": "1. Tieng Viet",
        "lang_en": "2. English",
        "lang_prompt": "Nhap lua chon (1-2, mac dinh 1): ",
        "features": "CHON CHUC NANG:",
        "menu_1": "1. [DATASET]  Xu ly anh tho (Raw -> Dataset)",
        "menu_1_sub": "   (Dung khi moi them anh vao folder raw_data)",
        "menu_2": "2. [VECTOR]   Tao du lieu AI (Dataset -> Embeddings)",
        "menu_2_sub": "   (Dung sau khi buoc 1 chay xong)",
        "menu_3": "3. [START]    Chay Face Swap (Webcam Real-time)",
        "menu_3_sub": "   (Che do man hinh doi: That vs Fake)",
        "menu_4": "4. [VIDEO]    Trich xuat anh tu video",
        "menu_4_sub": "   (Nhay file .mp4 vao folder video_data -> Tu tao Dataset)",
        "menu_5": "5. [RENDER]   Swap mat vao video co san",
        "menu_5_sub": "   (Dat file video input trong thu muc hien tai)",
        "menu_6": "6. [HEAD]     Fake toan bo dau (LivePortrait)",
        "menu_6_sub": "   (Cham hon, nhung phu ca dau thay vi chi mat)",
        "menu_7": "7. [MODEL]    Tu dong tai model theo cau hinh may",
        "menu_7_sub": "   (May yeu: goi nhe, may manh: goi day du)",
        "menu_0": "0. Thoat",
        "choice": "Nhap lua chon (0-7): ",
        "starting": ">>> Dang khoi dong: {script}...",
        "file_missing": "[LOI] Khong tim thay file: {script}",
        "script_error": "[LOI] Co loi khi chay script: {err}",
        "script_stopped": "[INFO] Da dung script.",
        "done_enter": ">>> Da xong! Nhan Enter de quay lai menu chinh...",
        "invalid": "Lua chon khong hop le!",
        "exit": "Da thoat.",
    },
    "en": {
               "done_enter": ">>> Đã xong! Nhấn Enter để quay lại menu chính...",
               "back_enter": "Nhấn Enter để quay lại menu...",
        "subtitle": "(Auto backend: CUDA / DirectML / CPU)",
        "pick_lang": "Choose language / Chon ngon ngu:",
        "lang_vi": "1. Tieng Viet",
        "lang_en": "2. English",
        "lang_prompt": "Select option (1-2, default 2): ",
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
               "done_enter": ">>> Done! Press Enter to return to the main menu...",
               "back_enter": "Press Enter to return to menu...",
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