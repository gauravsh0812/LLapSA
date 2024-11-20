import os
import math
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from scripts.dino_encoder import DinoVisionTower
    

def load_video(vis_path, num_frm=100):
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)
    total_num_frm = min(total_frame_num, num_frm)
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    a, H, W, _ = img_array.shape
    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--video_dir_path", required=True, help="Path to read the videos from.")
    parser.add_argument("--clip_feat_path", required=True, help="The output dir to save the features in.")
    args = parser.parse_args()
    return args


def load_and_stack_hidden_states(temp, video_id, 
                                 counter, vcgpt_features):

    # Iterate over each video ID
    hidden_states = []
    for i in range(counter):
        with open(os.path.join(temp, f"vcgpt_{video_id}_{i}.pkl"), 'rb') as f:
            hidden_state = pickle.load(f)
            hidden_states.append(hidden_state)

        stacked_states = torch.stack(hidden_states, dim=0)
        output_file = os.path.join(vcgpt_features, f"{video_id}.pkl")
        
        with open(output_file, 'wb') as f:
            pickle.dump(stacked_states, f)

def main():

    x = 0
    y = 10
    n = 0

    args = parse_args()
    video_dir_path = args.video_dir_path
    clip_feat_path = args.clip_feat_path
    vcgpt_features = os.path.join(clip_feat_path, "dino_features")
    temp = os.path.join(clip_feat_path, f"temp_{n}")
    os.makedirs(temp, exist_ok=True)
    os.makedirs(vcgpt_features, exist_ok=True)

    # Initialize the CLIP model
    dino = DinoVisionTower()
    all_videos = os.listdir(video_dir_path)
    all_videos = all_videos[x:y]

    for video_name in tqdm(all_videos):
        video_path = f"{video_dir_path}/{video_name}"
        video_id = video_name.split('.')[0]
        
        # try:
        frames = load_video(video_path)
        features = dino(frames)
        print("features shape: ", features.shape)

        break

        counter = 0    


        # try:
        for i in range(len(frames)):
            # features = dino(frames[i])
            # with open(f"{temp}/vcgpt_{video_id}_{i}.pkl", 'wb') as f:
            #     pickle.dump(features, f)
            
            counter +=1
            
        # print(counter, len(frames))
        # assert counter == len(frames) 
        # load_and_stack_hidden_states(temp, video_id, counter, vcgpt_features)
            
        # # clear the temp
        # for item in os.listdir(temp):
        #     item_path = os.path.join(temp, item)
        #     os.remove(item_path)

        # except Exception as e:
        #     print(f"Can't process {video_path}")

if __name__ == "__main__":
    main()  


# git pull; CUDA_VISIBLE_DEVICES=0 python scripts/save_vcgpt_features.py --video_dir_path /data/shared/gauravs/llapsa/vcgpt_clips --clip_feat_path /data/shared/gauravs/llapsa/longvu_videos