import numpy as np
import torch, sys, os

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

_VIDEO_EXTS = ('.mp4', '.mov', '.mkv', '.avi', '.webm')

args = load_args()


def _normalize_text(s: str) -> str:
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
                lang = line.split('Lang:', 1)[1].strip().lower()
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


def _compute_wer(hypothesis: str, reference: str):
    hyp_words = _normalize_text(hypothesis).split()
    gt_words = _normalize_text(reference).split()
    total_words = len(gt_words)
    if total_words == 0:
        return 0.0, 0, 0
    errors = levenshtein(hyp_words, gt_words)
    wer = errors / total_words
    return wer, errors, total_words


def _collect_video_paths(arg_obj):
    if arg_obj.fpath and arg_obj.input_dir:
        raise ValueError("Provide either --fpath or --input_dir, not both.")

    if arg_obj.fpath:
        if not os.path.isfile(arg_obj.fpath):
            raise FileNotFoundError(f"Input file not found: {arg_obj.fpath}")
        return [arg_obj.fpath]

    if arg_obj.input_dir:
        if not os.path.isdir(arg_obj.input_dir):
            raise FileNotFoundError(f"Input directory not found: {arg_obj.input_dir}")

        video_paths = []
        for root, _, files in os.walk(arg_obj.input_dir):
            for name in files:
                if name.lower().endswith(_VIDEO_EXTS):
                    video_paths.append(os.path.join(root, name))

        video_paths.sort()
        if not video_paths:
            raise FileNotFoundError(f"No video files found under directory: {arg_obj.input_dir}")
        return video_paths

    raise ValueError("Specify --fpath for single inference or --input_dir for directory inference.")


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

    pred_text = ' '.join(preds)
    return lang_id, pred_text


def load_models(arg_obj):
    model = build_model().to(arg_obj.device).eval()
    checkpoint = torch.load(arg_obj.ckpt_path, map_location=arg_obj.device)
    s = checkpoint["state_dict"]
    new_s = {}

    for k, v in s.items():
        if k.startswith("module."):
            new_s[k[7:]] = v
        else:
            new_s[k] = v

    model.load_state_dict(new_s)

    visual_encoder = build_visual_encoder().to(arg_obj.device).eval()
    s = torch.load(arg_obj.visual_encoder_ckpt_path, map_location=arg_obj.device)["state_dict"]
    new_s = {}
    for k, v in s.items():
        if "face_encoder" not in k:
            continue
        k = k.replace("module.face_encoder.", "")
        new_s[k] = v

    visual_encoder.load_state_dict(new_s)
    print("Following models are loaded successfully: \n{}\n{}".format(arg_obj.ckpt_path, arg_obj.visual_encoder_ckpt_path))

    return model, visual_encoder


def read_video(fpath, start=0, end=None):
    start *= 25
    start = max(start - 4, 0)

    if end is not None:
        end *= 25
        end += 4 # to read till end + 3
    else:
        end = 100000000000000000000000000 # some large finite num

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


if __name__ == '__main__':
    assert args.ckpt_path is not None, 'Specify a trained lip-reading checkpoint!'
    assert args.visual_encoder_ckpt_path is not None, 'Specify a trained feature extractor checkpoint!'

    try:
        video_paths = _collect_video_paths(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[Error] {exc}")
        sys.exit(1)

    model, visual_encoder = load_models(args)

    total = len(video_paths)
    progress = tqdm(video_paths, desc="Inference", unit="video")
    lang_hits = defaultdict(int)
    lang_wers = defaultdict(list)
    overall_wers = []
    total_errors = 0
    total_words = 0
    lang_errors = defaultdict(int)
    lang_words = defaultdict(int)

    for idx, video_path in enumerate(progress, start=1):
        progress.set_postfix_str(f"{idx}/{total}")
        progress.write("===================================================================")
        progress.write(f"[{idx}/{total}] Processing: {video_path}")

        try:
            faces = read_video(video_path, args.start, args.end)
        except Exception as exc:
            progress.write(f"[Error] Failed to read video {video_path}: {exc}")
            continue

        progress.write(f"Extracted frames from the input video: {faces.shape}")
        progress.write("Running inference...")
        try:
            lang_id, pred_text = run(faces, model, visual_encoder)
        except Exception as exc:
            progress.write(f"[Error] Inference failed for {video_path}: {exc}")
            del faces
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
            continue

        base, _ = os.path.splitext(video_path)
        ref_txt = base + '.txt'
        if os.path.exists(ref_txt):
            ref_text, ref_lang = _parse_reference_text(ref_txt)
        else:
            ref_text, ref_lang = (None, None)

        base_dir = os.path.basename(os.path.dirname(video_path))
        video_name = os.path.basename(video_path)
        display_name = f"{base_dir}/{video_name}" if base_dir else video_name

        progress.write("-------------------------------------------------------------------")
        progress.write("-------------------------------------------------------------------")
        progress.write(f"Video: {display_name}")
        target_lang = (ref_lang or args.lang_id or "unknown").strip() or "unknown"
        pred_lang = (lang_id or "unknown").strip() or "unknown"
        progress.write(f"Target language: {target_lang}")
        progress.write(f"Predicted language: {pred_lang}")
        if ref_text:
            progress.write(f"\n[Reference]: {ref_text}\n")
        progress.write(f"[Prediction]: {pred_text}")
        progress.write("-------------------------------------------------------------------")
        progress.write("-------------------------------------------------------------------")

        if ref_text:
            wer, errors, words = _compute_wer(pred_text, ref_text)
            progress.write(f"WER: {wer*100:.2f}%  (errors={errors}, words={words})")
            overall_wers.append(wer)
            lang_key = target_lang
            lang_wers[lang_key].append(wer)
            if target_lang != "unknown" and pred_lang == target_lang:
                lang_hits[target_lang] += 1
            total_errors += errors
            total_words += words
            lang_errors[lang_key] += errors
            lang_words[lang_key] += words
        else:
            progress.write("No reference .txt found alongside input video; skipping WER.")

        del faces
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()

    progress.close()
    print("-------------------------------------------------------------------")
    print(f"Requested language: {args.lang_id or 'unknown'}")
    if total_words > 0:
        global_wer = float(total_errors) / float(total_words) * 100
        print(f"Global WER: {global_wer:.2f}% (errors={total_errors}, words={total_words})")
        if lang_words:
            print("Global WER by language:")
            for lang in sorted(lang_words.keys()):
                l_errors = lang_errors.get(lang, 0)
                l_words = lang_words.get(lang, 0)
                l_wer = float(l_errors) / float(l_words) * 100 if l_words > 0 else 0.0
                hits = lang_hits.get(lang, 0)
                total_videos = len(lang_wers.get(lang, []))
                print(f"  {lang}: {l_wer:.2f}% (errors={l_errors}, words={l_words}, n={total_videos}, correct={hits}/{total_videos})")
    else:
        print("No reference transcripts found. WER summary unavailable.")
