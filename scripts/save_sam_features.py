import os
import math
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import SamModel, SamImageProcessor
import bitsandbytes as bnb

torch.cuda.empty_cache()

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

    return clip_imgs, total_num_frm


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq


def get_spatio_temporal_features(features, num_temporal_tokens=100):
    t, s, c = features.shape

    temporal_tokens = np.mean(features, axis=1)
    padding_size = num_temporal_tokens - t
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')

    spatial_tokens = np.mean(features, axis=0)
    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)

    return sp_features

def main():
    video_dir_path = "/data/shared/gauravs/llapsa/vcgpt_clips"
    clip_feat_path = "/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos"
    os.makedirs(clip_feat_path, exist_ok=True)

    sam_image_processor = SamImageProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50", torch_dtype=torch.float16)
    sam_model = SamModel.from_pretrained(
        "Zigeng/SlimSAM-uniform-50", 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,).cuda()   

    sam_model.eval()

    all_videos = os.listdir(video_dir_path)
    video_clip_features = {}
    counter = 0
    for video_name in tqdm(all_videos):
        video_path = f"{video_dir_path}/{video_name}"
        video_id = video_name.split('.')[0]
        if os.path.exists(f"{clip_feat_path}/{video_id}.pkl"):  # Check if the file is already processed
            continue
        # try:
        video,total_num_frm = load_video(video_path)
        for i in range(total_num_frm):
            clip = video[i,:,:,:]
            sam_tensor = sam_image_processor.preprocess(clip, return_tensors="pt")['pixel_values']
            sam_tensor = sam_tensor.half().cuda()

            # image_forward_outs = vision_tower(video_tensor, output_hidden_states=True)
            sam_forward_outs = sam_model(sam_tensor, output_hidden_states=True)

        break

        select_hidden_state_layer = -2
        select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
        batch_features = select_hidden_state[:, 1:]
        video_features[min_ind:max_ind] = batch_features.detach().cpu()

        video_clip_features[video_id] = get_spatio_temporal_features(video_features.numpy().astype("float16"))
        counter += 1

        # except Exception as e:
        #     print(f"Can't process {video_path}")

        break
        
        if counter % 512 == 0:  # Save after every 512 videos, update this number as per your requirements
            for key in video_clip_features.keys():
                features = video_clip_features[key]
                with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
                    pickle.dump(features, f)
            video_clip_features = {}
    exit()
    for key in video_clip_features.keys():
        features = video_clip_features[key]
        with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
            pickle.dump(features, f)


if __name__ == "__main__":
    main()