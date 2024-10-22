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

    return clip_imgs

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
    sam_preds = "/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/sam_preds"
    sam_iou = "/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/sam_iou"
    sam_hidden = "/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/sam_hidden_states"
    for i in [clip_feat_path,
              sam_preds,sam_iou,sam_hidden]:
        os.makedirs(i, exist_ok=True)

    sam_image_processor = SamImageProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50", torch_dtype=torch.float16)
    sam_model = SamModel.from_pretrained(
        "Zigeng/SlimSAM-uniform-50", 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,).cuda()   

    sam_model.eval()

    all_videos = os.listdir(video_dir_path)
    for video_name in tqdm(all_videos):
        video_path = f"{video_dir_path}/{video_name}"
        video_id = video_name.split('.')[0]
        if os.path.exists(f"{clip_feat_path}/{video_id}.pkl"):  # Check if the file is already processed
            continue
        # try:
        video = load_video(video_path)
        for i in range(len(video)):
            clip = video[i] #(224,224)
            sam_tensor = sam_image_processor.preprocess(clip, return_tensors="pt")['pixel_values'] 
            sam_tensor = sam_tensor.half().cuda() # (1,3,1024,1024)
            sam_forward_outs = sam_model(sam_tensor, output_hidden_states=True, return_dict=True)
            iou_score = sam_forward_outs.iou_scores #(1,1,3)
            pred_masks = sam_forward_outs.pred_masks  # torch.Size([1, 1, 3, 256, 256])
            hidden_states = sam_forward_outs.vision_hidden_states   

            print(hidden_states[-1].shape, hidden_states[-2].shape)

            with open(f"{sam_preds}/{video_id}.pkl", 'wb') as f:
                pickle.dump(pred_masks, f)
            with open(f"{sam_iou}/{video_id}.pkl", 'wb') as f:
                pickle.dump(iou_score, f)
            with open(f"{sam_hidden}/{video_id}.pkl", 'wb') as f:
                pickle.dump(hidden_states, f)
            
            break
            
            

if __name__ == "__main__":
    main()
