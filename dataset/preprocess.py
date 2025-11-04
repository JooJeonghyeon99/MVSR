import cv2
import numpy as np
import argparse
import os
import pickle
import subprocess
from scipy import signal
from glob import glob
from shutil import copy
from tqdm import tqdm  # [추가] 진행바

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video files for lipreading')
    parser.add_argument('--videos_folder', type=str, required=True, help='Folder containing unprocessed video files')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory for the processed dataset')
    parser.add_argument('--temp_dir', type=str, required=True, help='Directory for temporary files')
    parser.add_argument('--frame_rate', type=int, default=25, help='Frame rate of the video')

    parser.add_argument('--rank', type=int, default=0, help='Rank of the current process for distributed processing')
    parser.add_argument('--nshard', type=int, default=1, help='Total number of shards for distributed processing')
    args = parser.parse_args()
    return args


# python /mnt/aix7804/multivsr/dataset/preprocess.py --videos_folder /mnt/aix7804/multivsr/dataset/data/videos --data_root /mnt/aix7804/multivsr --temp_dir /mnt/aix7804/multivsr/dataset/data/temp --frame_rate 25 --rank 

args = parse_args()

# Append rank and shard to temp_dir to create unique directories for distributed processing
args.temp_dir = os.path.join(args.temp_dir, f"rank{args.rank}_of_{args.nshard}")

# Create the temporary directory if it doesn't exist
os.makedirs(args.temp_dir, exist_ok=True)


def crop_video(track):
    videofile = os.path.join(args.videos_folder, track.split('/')[-2] + '.mp4')
    temp_videofile = os.path.join(args.temp_dir, track.split('/')[-2] + '.mp4')

    # [추가] 원본 비디오가 없으면 조용히 스킵
    if not os.path.exists(videofile):
        print(f"[rank {args.rank}] skip - missing source video: {videofile}")
        return

    command = [
        'ffmpeg',
        '-i', videofile,
        '-r', str(args.frame_rate),
        '-y',  # Overwrite if exists
        temp_videofile
    ]
    # [개선] ffmpeg 반환코드 확인해서 실패 시 스킵
    ret = subprocess.call(command)
    if ret != 0:
        print(f"[rank {args.rank}] skip - ffmpeg failed for {videofile}")
        return
    
    videofile = temp_videofile

    if not os.path.exists(videofile):
        print(f"[rank {args.rank}] skip - temp video not created: {videofile}")
        return
    
    cropfile = track.replace('.pckl', '.mp4')
    if os.path.exists(cropfile):
        print(f"Cropped video file {cropfile} already exists")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vOut = cv2.VideoWriter(cropfile, fourcc, args.frame_rate, (224,224))
    if not vOut.isOpened():
        print(f"[rank {args.rank}] skip - failed to open output video {cropfile}")
        return

    track_file = track

    dets = {'x':[], 'y':[], 's':[]}
    try:
        with open(track_file, 'rb') as f:
            track_data = pickle.load(f)
    except Exception as exc:
        print(f"[rank {args.rank}] skip - failed to load track {track_file}: {exc}")
        vOut.release()
        return

    if not isinstance(track_data, dict) or 'bbox' not in track_data or 'frame' not in track_data:
        print(f"[rank {args.rank}] skip - invalid track format: {track_file}")
        vOut.release()
        return

    if len(track_data['bbox']) == 0 or len(track_data['frame']) == 0:
        print(f"[rank {args.rank}] skip - empty track data: {track_file}")
        vOut.release()
        return

    track = track_data

    for det in track['bbox']:

      dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
      dets['y'].append((det[1]+det[3])/2) # crop center x 
      dets['x'].append((det[0]+det[2])/2) # crop center y

    # Smooth detections
    try:
        dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
        dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
        dets['y'] = signal.medfilt(dets['y'],kernel_size=13)
    except ValueError as exc:
        print(f"[rank {args.rank}] skip - smoothing failed for {track_file}: {exc}")
        vOut.release()
        return

    
    frame_no_to_start = track['frame'][0]

    video_stream = cv2.VideoCapture(videofile)
    if not video_stream.isOpened():
        print(f"[rank {args.rank}] skip - cannot open video stream: {videofile}")
        vOut.release()
        return
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_no_to_start)

    # [추가] 프레임 범위에 tqdm 적용 - 각 rank마다 한 줄씩 표시
    frame_range = range(track['frame'][0], track['frame'][-1] + 1)
    for fidx, _ in enumerate(tqdm(frame_range,
                                  desc=f"frames r{args.rank}",
                                  unit="f",
                                  position=args.rank,
                                  leave=False)):
      if fidx >= len(dets['s']):
          print(f"[rank {args.rank}] warn - detection shorter than frame range for {videofile}")
          break
      cs  = 0.4

      bs  = dets['s'][fidx]   # Detection box size

      bsi = int(bs*(1+2*cs))  # Pad videos by this amount 

      ok, image = video_stream.read()
      if not ok or image is None:
          # [추가] 프레임 read 실패하면 해당 클립 조용히 종료
          print(f"[rank {args.rank}] warn - failed to read frame at idx {fidx}")
          break
 
      frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))

      my  = dets['y'][fidx]+bsi  # BBox center Y
      mx  = dets['x'][fidx]+bsi  # BBox center X

      face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]

      if face is not None and face.size != 0:
        vOut.write(cv2.resize(face,(224,224)))
      else:
        print("Empty face at video %d"%videofile)
        continue
    video_stream.release()

    vOut.release()
    
    print('Written %s'%cropfile)

if __name__ == "__main__":
    clips = list(glob(os.path.join(args.data_root, '*/*.pckl')))
    
    # Calculate the number of clips per shard
    clips_per_shard = int(np.ceil(len(clips) / args.nshard))
    # Calculate the start and end indices for this rank
    start_idx = args.rank * clips_per_shard
    end_idx = start_idx + clips_per_shard if args.rank < args.nshard - 1 else len(clips)
    # Get the subset of clips for this rank
    clips = clips[start_idx:end_idx]
    print(f"Rank {args.rank}: Processing {len(clips)} clips from index {start_idx} to {end_idx-1}")

    # [추가] 클립 진행률 tqdm
    for clip in tqdm(clips,
                     desc=f"clips r{args.rank}",
                     unit="clip",
                     position=args.rank):
        try:
            crop_video(clip)
        except Exception as exc:
            print(f"[rank {args.rank}] error - unexpected failure for {clip}: {exc}")



"""
python /mnt/aix7804/multivsr/dataset/preprocess.py \
--videos_folder /mnt/aix7804/multivsr/dataset/data/videos \
--data_root /mnt/aix7804/multivsr/dataset/data/metadata/multivsr \
--temp_dir /mnt/aix7804/multivsr/dataset/data/temp \
--frame_rate 25
"""
