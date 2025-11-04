import numpy as np
import torch, cv2, pickle, sys, os

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from collections import defaultdict

from models import build_model, build_visual_encoder

from dataloader import tokenizer
from config import load_args

from torch.amp import autocast
from search import beam_search

from decord import VideoReader

_VIDEO_EXTS = ('.mp4', '.mov', '.mkv', '.avi', '.webm')

args = load_args()


def _normalize_text(s: str) -> str:
	return s.strip().lower()


def _parse_reference_text(txt_path: str) -> str:
	"""Parse reference transcript from a .txt file.
	Prefers a line starting with 'Text:'; otherwise, concatenates WORD column.
	"""
	try:
		with open(txt_path, 'r', encoding='utf-8') as f:
			lines = [l.rstrip('\n') for l in f]
		for line in lines:
			if line.startswith('Text:'):
				return _normalize_text(line.split('Text:', 1)[1])
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
			return _normalize_text(' '.join(words))
	except Exception:
		pass
	return None


def _wer_details(ref_words, hyp_words):
	"""Compute S, D, I and WER using dynamic programming."""
	n = len(ref_words)
	m = len(hyp_words)
	dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
	for i in range(1, n + 1):
		c, S, D, I = dp[i - 1][0]
		dp[i][0] = (c + 1, S, D + 1, I)
	for j in range(1, m + 1):
		c, S, D, I = dp[0][j - 1]
		dp[0][j] = (c + 1, S, D, I + 1)
	for i in range(1, n + 1):
		for j in range(1, m + 1):
			c_sub, S_sub, D_sub, I_sub = dp[i - 1][j - 1]
			if ref_words[i - 1] == hyp_words[j - 1]:
				best = (c_sub, S_sub, D_sub, I_sub)
			else:
				best = (c_sub + 1, S_sub + 1, D_sub, I_sub)
			c_ins, S_ins, D_ins, I_ins = dp[i][j - 1]
			cand_ins = (c_ins + 1, S_ins, D_ins, I_ins + 1)
			c_del, S_del, D_del, I_del = dp[i - 1][j]
			cand_del = (c_del + 1, S_del, D_del + 1, I_del)
			best = min(best, cand_ins, cand_del, key=lambda x: (x[0], x[1] + x[2] + x[3]))
			dp[i][j] = best
	cost, S, D, I = dp[n][m]
	wer = cost / max(1, n)
	return wer, S, D, I, n


def _collect_video_paths(args):
	if args.fpath and args.input_dir:
		raise ValueError("Provide either --fpath or --input_dir, not both.")

	if args.fpath:
		if not os.path.isfile(args.fpath):
			raise FileNotFoundError(f"Input file not found: {args.fpath}")
		return [args.fpath]

	if args.input_dir:
		if not os.path.isdir(args.input_dir):
			raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

		video_paths = []
		for root, _, files in os.walk(args.input_dir):
			for name in files:
				if name.lower().endswith(_VIDEO_EXTS):
					video_paths.append(os.path.join(root, name))

		video_paths.sort()

		if not video_paths:
			raise FileNotFoundError(f"No video files found under directory: {args.input_dir}")

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

		print("★★★", feats.shape)
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

		if i==0:
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

	try:
		video_paths = _collect_video_paths(args)
	except (ValueError, FileNotFoundError) as exc:
		print(f"[Error] {exc}")
		sys.exit(1)

	model, visual_encoder = load_models(args)

	total = len(video_paths)
	progress = tqdm(video_paths, desc="Inference", unit="video", ascii=('□', '■'))
	overall_wers = []
	lang_wers = defaultdict(list)

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
		ref_text = _parse_reference_text(ref_txt) if os.path.exists(ref_txt) else None

		progress.write("-------------------------------------------------------------------")
		progress.write("-------------------------------------------------------------------")
		progress.write(f"Video: {video_path}")
		progress.write(f"Language: {lang_id}")
		if ref_text:
			progress.write(f"Reference: {ref_text}")
		progress.write(f"Prediction: {pred_text}")
		progress.write("-------------------------------------------------------------------")
		progress.write("-------------------------------------------------------------------")

		if ref_text:
			ref_words = _normalize_text(ref_text).split()
			hyp_words = _normalize_text(pred_text).split()
			wer, S, D, I, N = _wer_details(ref_words, hyp_words)
			progress.write(f"WER: {wer*100:.2f}%  (S={S}, D={D}, I={I}, N={N})")
			overall_wers.append(wer)
			lang_key = (lang_id or "unknown").strip() or "unknown"
			lang_wers[lang_key].append(wer)
		else:
			progress.write("No reference .txt found alongside input video; skipping WER.")

		del faces
		if args.device.startswith('cuda'):
			torch.cuda.empty_cache()

	progress.close()
	print("===================================================================")
	if overall_wers:
		avg_wer = float(np.mean(overall_wers)) * 100
		print(f"Average WER: {avg_wer:.2f}% (videos with refs: {len(overall_wers)})")
		if lang_wers:
			print("Average WER by language:")
			for lang, values in sorted(lang_wers.items()):
				lang_avg = float(np.mean(values)) * 100
				print(f"  {lang}: {lang_avg:.2f}% (n={len(values)})")
	else:
		print("No reference transcripts found; WER summary unavailable.")
