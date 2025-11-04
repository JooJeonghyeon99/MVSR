import argparse
import os
import pickle
import subprocess
from shutil import copy

import cv2
import numpy as np
from scipy import signal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess a single video clip for lipreading (single-sample version of preprocess.py)"
    )
    parser.add_argument(
        "--videos_folder",
        type=str,
        required=True,
        help="Folder containing unprocessed video files (video_id.mp4)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory with input metadata (contains video_id/clip_id.pckl and .txt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for preprocessed files (will contain video_id/clip_id.{pckl,txt,mp4,wav})",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        required=True,
        help="Directory for temporary files",
    )
    parser.add_argument(
        "--frame_rate",
        type=int,
        default=25,
        help="Frame rate of the video",
    )
    
    # Single clip selection
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Relative id like 'video_id/clip_id' to process a single clip",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Full path to a single .pckl track file to process",
    )
    
    # Optional: keep rank/nshard for compatibility (but use defaults)
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank of the current process (for compatibility with original)",
    )
    parser.add_argument(
        "--nshard",
        type=int,
        default=1,
        help="Total number of shards (for compatibility with original)",
    )
    
    return parser.parse_args()

args = parse_args()

# Create unique temp_dir for this rank (same as original)
args.temp_dir = os.path.join(args.temp_dir, f"rank{args.rank}_of_{args.nshard}")
os.makedirs(args.temp_dir, exist_ok=True)


def crop_video(track):
    """
    Process one clip: copy .pckl and .txt to output_dir, generate .mp4 and .wav there.
    """
    # Extract video_id and clip_id from track path
    # track = data_root/video_id/clip_id.pckl
    video_id = track.split("/")[-2]
    clip_id = os.path.splitext(os.path.basename(track))[0]  # e.g., "00002"
    
    # Create output directory structure
    output_video_dir = os.path.join(args.output_dir, video_id)
    os.makedirs(output_video_dir, exist_ok=True)
    
    # === 1. Copy .pckl and .txt to output directory ===
    src_pckl = track
    dst_pckl = os.path.join(output_video_dir, f"{clip_id}.pckl")
    copy(src_pckl, dst_pckl)
    
    src_txt = track.replace(".pckl", ".txt")
    dst_txt = os.path.join(output_video_dir, f"{clip_id}.txt")
    if os.path.exists(src_txt):
        copy(src_txt, dst_txt)
    
    # === 2. Process video ===
    videofile = os.path.join(args.videos_folder, video_id + ".mp4")
    temp_videofile = os.path.join(args.temp_dir, video_id + ".mp4")

    # Re-encode to target frame rate
    command = [
        "ffmpeg",
        "-i",
        videofile,
        "-r",
        str(args.frame_rate),
        "-y",
        temp_videofile,
    ]
    subprocess.call(command)

    videofile = temp_videofile

    if not os.path.exists(videofile):
        raise ValueError(f"temp video file {videofile} could not be created")

    # Output .mp4 in output_dir
    cropfile = os.path.join(output_video_dir, f"{clip_id}.mp4")
    if os.path.exists(cropfile):
        print(f"Cropped video file {cropfile} already exists")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vOut = cv2.VideoWriter(cropfile, fourcc, args.frame_rate, (224, 224))

    dets = {"x": [], "y": [], "s": []}
    with open(track, "rb") as f:
        track_data = pickle.load(f)

    for det in track_data["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)  # crop center y
        dets["x"].append((det[0] + det[2]) / 2)  # crop center x

    # Smooth detections
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

    frame_no_to_start = track_data["frame"][0]

    video_stream = cv2.VideoCapture(videofile)
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_no_to_start)
    
    for fidx, frame in enumerate(
        range(track_data["frame"][0], track_data["frame"][-1] + 1)
    ):
        cs = 0.4
        bs = dets["s"][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount

        image = video_stream.read()[1]

        frame = np.pad(
            image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110)
        )

        my = dets["y"][fidx] + bsi  # BBox center Y
        mx = dets["x"][fidx] + bsi  # BBox center X

        face = frame[
            int(my - bs) : int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
        ]

        vOut.write(cv2.resize(face, (224, 224)))
    
    video_stream.release()
    vOut.release()

    # ========== EXTRACT AND CROP AUDIO FILE ==========
    # Extract audio from temp video, trim to clip range, save as .wav
    try:
        audiotmp = os.path.join(args.temp_dir, "audio.wav")
        audiostart = track_data["frame"][0] / args.frame_rate
        audioend = (track_data["frame"][-1] + 1) / args.frame_rate

        # Extract full audio from temp video
        cmd_extract = [
            "ffmpeg",
            "-y",
            "-i",
            videofile,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            audiotmp,
        ]
        subprocess.call(cmd_extract)

        # Trim to clip range
        out_wav = cropfile.replace(".mp4", ".wav")
        cmd_trim = [
            "ffmpeg",
            "-y",
            "-i",
            audiotmp,
            "-ss",
            f"{audiostart:.3f}",
            "-to",
            f"{audioend:.3f}",
            out_wav,
        ]
        subprocess.call(cmd_trim)
    except Exception as e:
        print(f"[WARN] Audio extraction failed: {e}")

    print("Written %s" % cropfile)


if __name__ == "__main__":
    # Select single clip
    track = None
    if args.track is not None:
        track = args.track
    elif args.id is not None:
        # id format: video_id/clip_id
        track = os.path.join(args.data_root, args.id + ".pckl")
    else:
        raise SystemExit(
            "Specify either --track (full path to .pckl) or --id (video_id/clip_id)"
        )

    if not os.path.exists(track):
        raise FileNotFoundError(f"Track file not found: {track}")

    print(f"Processing single track: {track}")
    crop_video(track)
    print(f"\n✅ Preprocessing complete!")
    print(f"Output directory: {args.output_dir}")

"""
Example usage:

python /mnt/aix7804/multivsr/dataset/preprocess_sample.py \
  --videos_folder /mnt/aix7804/multivsr/dataset/data/videos \
  --data_root /mnt/aix7804/multivsr/dataset/data/metadata/multivsr \
  --output_dir /mnt/aix7804/multivsr/dataset/data/videos_preprocessed \
  --temp_dir /mnt/aix7804/multivsr/dataset/data/temp \
  --id  eXMRxhanW2E/00004 \
  --frame_rate 25

This will create:
  /mnt/aix7804/multivsr/dataset/data/videos_preprocessed/05fifNuuB7I/00002.pckl  (copied)
  /mnt/aix7804/multivsr/dataset/data/videos_preprocessed/05fifNuuB7I/00002.txt   (copied)
  /mnt/aix7804/multivsr/dataset/data/videos_preprocessed/05fifNuuB7I/00002.mp4   (generated)
  /mnt/aix7804/multivsr/dataset/data/videos_preprocessed/05fifNuuB7I/00002.wav   (generated)
"""