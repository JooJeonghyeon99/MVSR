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

    print("-------------------------------------------------------------------")
    print("-------------------------------------------------------------------")
    print(f"Video: {display_name}")
    target_lang = (ref_lang or args.lang_id or "unknown").strip() or "unknown"
    pred_lang = (lang_id or "unknown").strip() or "unknown"
    print(f"Target language: {target_lang}")
    print(f"Predicted language: {pred_lang}")
    if ref_text:
        print(f"\n[Reference]: {ref_text}\n")
    print(f"[Prediction]: {pred_text}")
    match = target_lang != "unknown" and pred_lang == target_lang
    print(f"Language match: {'correct' if match else 'incorrect'}")
    if ref_text:
        wer, errors, words = _compute_wer(pred_text, ref_text)
        print(f"WER: {wer*100:.2f}%  (errors={errors}, words={words})")
        print(f"Language WER ({target_lang}): {wer*100:.2f}%")
    else:
        print("No reference .txt found alongside input video; skipping WER.")
    print("-------------------------------------------------------------------")
    print("-------------------------------------------------------------------")
