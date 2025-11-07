import argparse
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Repackage mTEDx pywork outputs into metadata-style folders.")
    parser.add_argument(
        "--pywork_dir",
        type=str,
        default="/mnt/aix7804/multivsr/preprocess/mTEDx_results/pywork",
        help="Directory containing SyncNet pywork outputs (one subfolder per clip).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/aix7804/multivsr/dataset/data/metadata/mtedx",
        help="Destination root where video-id folders will be created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the destination directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of clips to convert (useful for smoke tests).",
    )
    return parser.parse_args()


def _alias_numpy_core():
    """Allow loading pickles that were created with numpy 2.x."""
    import sys

    if "numpy._core" not in sys.modules:
        try:
            import numpy.core as np_core

            sys.modules["numpy._core"] = np_core
            import numpy.core._multiarray_umath as umath

            sys.modules["numpy._core._multiarray_umath"] = umath
        except ImportError:
            pass


def _resolve_track_dict(track_data):
    """
    Extract the actual {'frame', 'bbox'} dict from various pywork pickle formats.
    Chooses the longest available track.
    """
    candidates = []
    if isinstance(track_data, dict):
        candidates.append(track_data)
    elif isinstance(track_data, list):
        for entry in track_data:
            if not isinstance(entry, dict):
                continue
            if "track" in entry and isinstance(entry["track"], dict):
                candidates.append(entry["track"])
            else:
                candidates.append(entry)

    best = None
    best_len = -1
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if "bbox" not in candidate or "frame" not in candidate:
            continue
        bbox = candidate["bbox"]
        length = len(bbox)
        if length > best_len:
            best = candidate
            best_len = length
    return best


def _clip_parts(clip_id):
    if "_" not in clip_id:
        return clip_id, "00000"
    video_id, segment = clip_id.rsplit("_", 1)
    return video_id, segment


def _serialize_track(track):
    frames = track["frame"]
    if hasattr(frames, "tolist"):
        frames = [int(v) for v in frames.tolist()]
    else:
        frames = [int(v) for v in frames]

    bbox = np.asarray(track["bbox"])

    return {"frame": frames, "bbox": bbox}


def main():
    args = parse_args()
    _alias_numpy_core()

    pywork_dir = Path(args.pywork_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    track_files = sorted(pywork_dir.glob("*/tracks.pckl"))
    if not track_files:
        print(f"No tracks.pckl files found under {pywork_dir}")
        return

    if args.limit is not None:
        track_files = track_files[: args.limit]

    copied = 0
    skipped = 0

    for track_file in tqdm(track_files, desc="clips", unit="clip"):
        clip_id = track_file.parent.name
        video_id, segment_id = _clip_parts(clip_id)
        dest_dir = output_dir / video_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_pckl = dest_dir / f"{segment_id}.pckl"
        dest_mp4 = dest_dir / f"{segment_id}.mp4"

        if dest_pckl.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            with open(track_file, "rb") as f:
                track_data = pickle.load(f)
        except Exception as exc:
            print(f"Skip {track_file} - failed to load pickle: {exc}")
            skipped += 1
            continue

        track = _resolve_track_dict(track_data)
        if track is None:
            print(f"Skip {track_file} - no usable track found")
            skipped += 1
            continue

        payload = _serialize_track(track)

        with open(dest_pckl, "wb") as f:
            pickle.dump(payload, f)

        src_mp4 = track_file.with_name("tracks.mp4")
        if src_mp4.exists():
            shutil.copy2(src_mp4, dest_mp4)
        else:
            print(f"Warn: {src_mp4} missing, only pickle exported.")

        dest_txt = dest_dir / f"{segment_id}.txt"
        if args.overwrite or not dest_txt.exists():
            dest_txt.write_text(clip_id, encoding="utf-8")

        copied += 1

    print(f"Done. Converted {copied} clips (skipped {skipped}). Output root: {output_dir}")


if __name__ == "__main__":
    main()


