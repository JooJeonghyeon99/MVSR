import json
import numpy as np
import torch, cv2, pickle, sys, os
from datetime import datetime

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

from models import build_model, build_visual_encoder

from dataloader import tokenizer
from config import load_args

from torch.amp import autocast
from search import beam_search
from test_scores import levenshtein

from decord import VideoReader

SEPARATOR = "=" * 60
_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm")

args = load_args()


def _normalize_text(s: str) -> str: #원래는 그냥 대체용이엇는데 손봐야 할듯 
    return s.strip().lower()


def _parse_reference_text(txt_path: str):
    text = None
    lang = None
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [l.rstrip('\n') for l in f]
        for line in lines:
            if line.startswith('Text:'):
                text = _normalize_text(line.split('Text:', 1)[1])
            elif line.startswith('Lang:'):
                lang = _normalize_text(line.split('Lang:', 1)[1])
        if text:
            return text, lang

        words = []
        in_table = False
        for line in lines:
            if line.strip().startswith('WORD'):
                in_table = True
                continue
            if in_table and line.strip():
                parts = [p.strip() for p in line.split(',')]
                if parts:
                    words.append(parts[0])
        if words:
            return _normalize_text(' '.join(words)), lang
    except Exception:
        pass
    return None, lang


def forward_pass(model, encoder_output, src_mask, start_symbol):

    beam_outs, beam_scores = beam_search(
            model=model,
            bos_index=start_symbol,
            eos_index=tokenizer.eot,
            max_output_length=args.max_decode_len,
            pad_index=0,
            encoder_output=encoder_output,
            src_mask=src_mask,
            size=args.beam_size,
            n_best=args.beam_size,
        )

    return beam_outs, beam_scores


def run(faces, model, visual_encoder):
    chunk_frames = args.chunk_size * 25
    preds = []
    if args.lang_id is not None:
        lang_id = args.lang_id
    else:
        lang_id = None

    for i in range(0, faces.size(2), chunk_frames):
        chunk_faces = faces[:, :, i:i+chunk_frames]
        chunk_faces = chunk_faces.to(args.device)
        feats = visual_encoder(chunk_faces)

        src_mask = torch.ones(1, 1, feats.shape[1]).long().to(args.device)

        encoder_output, src_mask = model.encode(feats, src_mask)

        with torch.no_grad():
            with autocast('cuda', enabled=args.fp16):
                start_symbol = [50258]
                if args.lang_id is not None:
                    start_symbol.append(tokenizer.encode(f"<|{args.lang_id}|>")[0])

                beam_outs, beam_scores = forward_pass(model, encoder_output, src_mask, start_symbol)
                out = beam_outs[0][0]

        pred = tokenizer.decode(out.cpu().numpy().tolist()[:-1]).strip().lower()

        pred = pred.replace("<|transcribe|><|notimestamps|>", " ")

        if i == 0:
            lang_id = pred[2:4]
        pred = pred[7:].strip()
        preds.append(pred)

    full_pred = ' '.join(preds)

    return lang_id, full_pred


def load_models(args):
    model = build_model().to(args.device).eval()
    checkpoint = torch.load(args.ckpt_path, map_location=args.device)
    s = checkpoint["state_dict"]
    new_s = {}

    for k, v in s.items():
        if k.startswith("module."):
            new_s[k[7:]] = v
        else:
            new_s[k] = v

    model.load_state_dict(new_s)

    visual_encoder = build_visual_encoder().to(args.device).eval()
    s = torch.load(args.visual_encoder_ckpt_path, map_location=args.device)["state_dict"]
    new_s = {}
    for k, v in s.items():
        if "face_encoder" not in k:
            continue
        k = k.replace("module.face_encoder.", "")
        new_s[k] = v

    visual_encoder.load_state_dict(new_s)
    print("Following models are loaded successfully: \n{}\n{}".format(args.ckpt_path, args.visual_encoder_ckpt_path))

    return model, visual_encoder


def read_video(fpath, start=0, end=None):
    start *= 25
    start = max(start - 4, 0)

    if end is not None:
        end *= 25
        end += 4
    else:
        end = 100000000000000000000000000

    with open(fpath, 'rb') as f:
        video_stream = VideoReader(f, width=160, height=160)

        end = min(end, len(video_stream))

        frames = video_stream.get_batch(list(range(int(start), int(end)))).asnumpy().astype(np.float32)
    frames = torch.FloatTensor(frames).to(args.device).unsqueeze(0)
    frames /= 255.
    frames = frames.permute(0, 4, 1, 2, 3)
    crop_x = (frames.size(3) - 96) // 2
    crop_y = (frames.size(4) - 96) // 2
    faces = frames[:, :, :, crop_x:crop_x + 96 ,
                        crop_y:crop_y + 96]

    return faces


def _collect_videos(root_dir: str):
    video_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(_VIDEO_EXTS):
                video_paths.append(os.path.join(dirpath, name))
    video_paths.sort()
    return video_paths


if __name__ == '__main__':
    assert args.ckpt_path is not None, 'Specify a trained lip-reading checkpoint!'
    assert args.visual_encoder_ckpt_path is not None, 'Specify a trained feature extractor checkpoint!'

    if not args.input_dir and not args.fpath:
        print("[Error] Provide --input_dir for folder inference or --fpath for a single video.")
        sys.exit(1)

    if args.input_dir:
        video_paths = _collect_videos(args.input_dir)
        if not video_paths:
            print(f"[Error] No videos found under {args.input_dir}")
            sys.exit(1)
    else:
        if not os.path.isfile(args.fpath):
            print(f"[Error] Input file not found: {args.fpath}")
            sys.exit(1)
        video_paths = [args.fpath]

    total_videos = len(video_paths)

    model, visual_encoder = load_models(args)

    total_wer = 0
    num_words = 0
    total_samples = 0
    successful_runs = 0
    lang_stats = defaultdict(lambda: {
        "errors": 0,
        "words": 0,
        "samples": 0,
        "hits": 0,
    })

    metrics_path = None
    if args.metrics_out:
        run_time = datetime.now()
        provided_path = os.path.expanduser(args.metrics_out)

        metrics_label = None
        metrics_root = provided_path

        if provided_path.lower().endswith('.jsonl'):
            metrics_root = os.path.dirname(provided_path)
            stem = os.path.splitext(os.path.basename(provided_path))[0]
            if stem:
                metrics_label = stem

        if not metrics_root:
            metrics_root = '.'

        date_dir = run_time.strftime("%Y-%m-%d")
        time_dir = run_time.strftime("%H-%M-%S")
        metrics_dir = os.path.join(metrics_root, date_dir)
        os.makedirs(metrics_dir, exist_ok=True)

        file_name = f"{metrics_label}.jsonl" if metrics_label else f"{time_dir}.jsonl"
        metrics_path = os.path.join(metrics_dir, file_name)
        print(f"[Metrics] Writing per-video logs to {metrics_path}")

    progress = tqdm(enumerate(video_paths, start=1), total=total_videos, desc="Inference", unit="video")

    for index, video_path in progress:
        total_samples += 1
        base, _ = os.path.splitext(video_path)
        ref_txt = base + '.txt'
        if os.path.exists(ref_txt):
            ref_text, ref_lang = _parse_reference_text(ref_txt)
        else:
            ref_text, ref_lang = (None, None)

        try:
            faces = read_video(video_path, args.start, args.end)
        except Exception as exc:
            progress.write(f"[Error] Failed to read video {video_path}: {exc}")
            continue

        progress.write(f"[{index}/{total_videos}] Processing: {video_path}")
        progress.write(f"Extracted frames: {faces.shape}")

        try:
            lang_id, pred_text = run(faces, model, visual_encoder)
        except Exception as exc:
            progress.write(f"[Error] Inference failed for {video_path}: {exc}")
            del faces
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
            continue

        display_name = os.path.relpath(video_path, args.input_dir) if args.input_dir else os.path.basename(video_path)
        target_lang = (ref_lang or args.lang_id or "unknown").strip() or "unknown"
        pred_lang = (lang_id or "unknown").strip() or "unknown"

        progress.write(SEPARATOR)
        progress.write(f"Video: {display_name}")
        progress.write(f"Target language: {target_lang}")
        progress.write(f"Predicted language: {pred_lang}")
        if ref_text:
            progress.write(f"Reference: {ref_text}")
        progress.write(f"Prediction: {pred_text}")

        video_errors = None
        video_words = None
        video_wer = None

        lang_entry = lang_stats[target_lang]
        lang_entry["samples"] += 1

        if target_lang != "unknown" and pred_lang == target_lang:
            lang_entry["hits"] += 1

        if ref_text:
            pred = pred_text.strip().lower()
            gt = ref_text.strip().lower()

            video_errors = levenshtein(pred.split(), gt.split())
            video_words = len(gt.split())
            total_wer += video_errors
            num_words += video_words

            lang_entry["errors"] += video_errors
            lang_entry["words"] += video_words

            video_wer = video_errors / video_words if video_words else 0.0
            progress.write(f"WER: {video_wer*100:.2f}%  (errors={video_errors}, words={video_words})")
        else:
            progress.write("WER: N/A (no reference transcript)")

        progress.write(SEPARATOR)

        metrics_entry = {
            "video": display_name,
            "target_lang": target_lang,
            "pred_lang": pred_lang,
            "prediction": pred_text,
            "reference": ref_text,
            "has_reference": bool(ref_text),
            "errors": video_errors,
            "words": video_words,
            "wer": video_wer,
        }

        if metrics_path:
            with open(metrics_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics_entry, ensure_ascii=False) + '\n')

        successful_runs += 1
        del faces
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()

    progress.close()

    if total_videos:
        print(f"Processed {successful_runs}/{total_videos} videos successfully.")

    overall = None
    if num_words:
        overall = total_wer / num_words
        print(f"Overall WER: {overall*100:.2f}% (errors={total_wer}, words={num_words})")
    else:
        print("Overall WER: N/A (no references)")

    if total_samples:
        total_hits = sum(stats["hits"] for stats in lang_stats.values())
        accuracy = total_hits / total_samples * 100
    else:
        accuracy = None

    if lang_stats:
        print("Language breakdown:")
        for lang in sorted(lang_stats.keys()):
            stats = lang_stats[lang]
            words = stats["words"]
            errors = stats["errors"]
            hits = stats["hits"]
            samples = stats["samples"]

            if words:
                lang_wer = errors / words * 100
                wer_str = f"WER {lang_wer:.2f}% (errors={errors}, words={words})"
            else:
                wer_str = "WER N/A"

            if samples:
                lang_acc = hits / samples * 100
                acc_str = f"accuracy {hits}/{samples})"
            else:
                acc_str = "accuracy N/A"

            print(f"  {lang}: {wer_str}, {acc_str}")

    if metrics_path:
        summary_entry = {
            "video": "__overall__",
            "overall_wer": overall,
            "total_errors": total_wer if num_words else None,
            "total_words": num_words if num_words else None,
            "total_videos": total_videos,
            "successful_videos": successful_runs,
            "language_accuracy": accuracy,
        }
        with open(metrics_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary_entry, ensure_ascii=False) + '\n')
