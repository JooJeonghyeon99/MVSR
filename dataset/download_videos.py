import argparse
import os, subprocess
from tqdm import tqdm
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import time, random

warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["LC_ALL"] = "en_US.utf-8"
os.environ["LANG"] = "en_US.utf-8"

parser = argparse.ArgumentParser(description="Code to download the videos from the input filelist of ids")
parser.add_argument('--file', type=str, required=True, help="Path to the input filelist of ids")
parser.add_argument('--video_root', type=str, required=True, help="Path to the directory to save the videos")
args = parser.parse_args()


def is_valid_video(file_path):
    """Check if video file exists and playable (duration only)"""
    if not os.path.exists(file_path):
        return False
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        duration = float(result.stdout.strip())
        if duration <= 0:
            return False
    except Exception:
        return False
    return True


def mp_handler(vid, result_dir):
    """Handles downloading one video with retries and backoff"""
    try:
        vid = vid.strip()
        if not vid:
            return

        video_link = f"https://www.youtube.com/watch?v={vid}"
        output_fname = os.path.join(result_dir, f"{vid}.mp4")
        cookie_file = "/mnt/aix7804/multivsr/dataset/filelists/cookies.txt"

        base_sleep = 3.0
        max_retries = 3

        # 이미 존재하고 유효하면 스킵
        if os.path.exists(output_fname) and is_valid_video(output_fname):
            return

        for attempt in range(1, max_retries + 1):
            cmd = (
                f'yt-dlp --geo-bypass --force-overwrites '
                f'--cookies "{cookie_file}" '
                f'-f "bv*+ba/b" -o "{output_fname}" "{video_link}"'
            )

            subprocess.run(cmd, shell=True)

            # 랜덤 sleep + exponential backoff
            sleep_time = base_sleep * random.uniform(0.8, 1.4) * (2 ** (attempt - 1))
            time.sleep(sleep_time)

            if is_valid_video(output_fname):
                break
            else:
                print(f"[Retry {attempt}] Invalid or failed: {vid}")
                if os.path.exists(output_fname):
                    os.remove(output_fname)

        if not is_valid_video(output_fname):
            print(f"[Failed after {max_retries} retries] {vid}")

    except KeyboardInterrupt:
        exit(0)
    except Exception:
        traceback.print_exc()


def download_data(args):
    """Main downloader"""
    with open(args.file, 'r', encoding='utf-8') as f:
        filelist = [x.strip() for x in f.readlines() if x.strip()]

    print(f"Total videos to download: {len(filelist)}")
    os.makedirs(args.video_root, exist_ok=True)

    max_workers = 5  # 병렬 다운로드 개수

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(mp_handler, vid, args.video_root) for vid in filelist]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            pass


if __name__ == '__main__':
    download_data(args)
