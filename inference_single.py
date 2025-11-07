import json
import numpy as np
import torch, cv2, pickle, sys, os

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob

from models import build_model, build_visual_encoder

from dataloader import tokenizer
from config import load_args

from torch.amp import autocast
from search import beam_search
from test_scores import levenshtein

from decord import VideoReader

SEPARATOR = "=" * 60

args = load_args()

total_wer = 0
num_words = 0


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
    assert args.fpath is not None, 'Single-file inference expects --fpath to be set.'

    model, visual_encoder = load_models(args)

    faces = read_video(args.fpath, args.start, args.end)
    print("Extracted frames from the input video: ", faces.shape)

    print("Running inference...")
    lang_id, pred_text = run(faces, model, visual_encoder)

    base_dir = os.path.basename(os.path.dirname(args.fpath))
    video_name = os.path.basename(args.fpath)
    display_name = f"{base_dir}/{video_name}" if base_dir else video_name

    base_path, _ = os.path.splitext(args.fpath)
    ref_txt = base_path + '.txt'
    if os.path.exists(ref_txt):
        ref_text, ref_lang = _parse_reference_text(ref_txt)
    else:
        ref_text, ref_lang = (None, None)

    print(SEPARATOR)
    print(f"Video: {display_name}")
    target_lang = (ref_lang or args.lang_id or "unknown").strip() or "unknown"
    print(f"Target language: {target_lang}")
    print(f"Predicted language: {lang_id}")
    if ref_text:
        print(f"Reference: {ref_text}")
    print(f"Prediction: {pred_text}")

    video_errors = None
    video_words = None
    video_wer = None

    if ref_text:
        pred = pred_text.strip().lower()
        gt = ref_text.strip().lower()

        video_errors = levenshtein(pred.split(), gt.split())
        video_words = len(gt.split())
        total_wer += video_errors
        num_words += video_words

        video_wer = video_errors / video_words if video_words else 0.0
        print(f"WER: {video_wer*100:.2f}%  (errors={video_errors}, words={video_words})")
    else:
        print("WER: N/A (no reference transcript)")

    print(SEPARATOR)

    metrics_entry = {
        "video": display_name,
        "target_lang": target_lang,
        "pred_lang": (lang_id or "unknown").strip() or "unknown",
        "prediction": pred_text,
        "reference": ref_text,
        "has_reference": bool(ref_text),
        "errors": video_errors,
        "words": video_words,
        "wer": video_wer,
    }

    if args.metrics_out:
        metrics_dir = os.path.dirname(args.metrics_out)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(args.metrics_out, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics_entry, ensure_ascii=False) + '\n')

if num_words:
    overall = total_wer / num_words
    print(f"Overall WER: {overall*100:.2f}% (errors={total_wer}, words={num_words})")
else:
    print("Overall WER: N/A (no references)")
