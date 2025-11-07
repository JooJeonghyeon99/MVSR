import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill metadata txt files with transcriptions from e-mvsr label files."
    )
    parser.add_argument(
        "--tsv_path",
        type=str,
        default="/mnt/aix7804/e-mvsr/labels/fr/test.tsv",
        help="TSV file that lists video paths (first usable row after the leading '/').",
    )
    parser.add_argument(
        "--wrd_path",
        type=str,
        default="/mnt/aix7804/e-mvsr/labels/fr/test.wrd",
        help="WRD file containing line-aligned transcriptions.",
    )
    parser.add_argument(
        "--metadata_root",
        type=str,
        default="/mnt/aix7804/multivsr/dataset/data/metadata/mtedx",
        help="Root directory containing <video_id>/<segment_id>.txt files.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Encoding used for input and output text files.",
    )
    parser.add_argument(
        "--create_missing",
        action="store_true",
        help="Create metadata directories/files if they do not already exist.",
    )
    return parser.parse_args()


def load_transcripts(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        return [line.rstrip("\n") for line in f]


def load_tsv_rows(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    if rows and len(rows[0]) == 1 and rows[0][0].strip() == "/":
        rows = rows[1:]
    return rows


def clip_parts_from_path(path_str):
    clip_stem = Path(path_str).stem
    if "_" not in clip_stem:
        return clip_stem, "00000"
    return clip_stem.rsplit("_", 1)


def main():
    args = parse_args()

    transcripts = load_transcripts(args.wrd_path, args.encoding)
    rows = load_tsv_rows(args.tsv_path, args.encoding)

    if len(rows) != len(transcripts):
        raise ValueError(
            f"Row mismatch: TSV has {len(rows)} usable lines whereas WRD has {len(transcripts)}."
        )

    metadata_root = Path(args.metadata_root)
    written = 0
    missing = 0

    for idx, (row, text) in enumerate(zip(rows, transcripts), start=1):
        if len(row) < 2:
            continue
        _, video_path, *rest = row
        video_id, segment_id = clip_parts_from_path(video_path)
        dest_dir = metadata_root / video_id
        dest_file = dest_dir / f"{segment_id}.txt"

        if not dest_dir.exists():
            if args.create_missing:
                dest_dir.mkdir(parents=True, exist_ok=True)
            else:
                print(f"[{idx}] skip - missing directory {dest_dir}")
                missing += 1
                continue

        if not dest_file.exists() and not args.create_missing:
            print(f"[{idx}] skip - missing file {dest_file}")
            missing += 1
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file.write_text(text + "\n", encoding=args.encoding)
        written += 1

    print(f"Done. Updated {written} transcripts, skipped {missing}. Output root: {metadata_root}")


if __name__ == "__main__":
    main()
