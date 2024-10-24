import os
import math
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import (
    CLIPVisionModel, 
    CLIPImageProcessor, 
    SamModel, SamImageProcessor)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--video_dir_path", required=True, help="Path to read the videos from.")
    parser.add_argument("--clip_feat_path", required=True, help="The output dir to save the features in.")
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    video_dir_path = args.video_dir_path
    clip_feat_path = args.clip_feat_path
    vcgpt_features = os.path.join(clip_feat_path, "vcgpt_features")
    sam_hidden_states = os.path.join(clip_feat_path, "sam_hidden_states")
    sam_preds = os.path.join(clip_feat_path, "sam_preds")
    sam_iou = os.path.join(clip_feat_path, "sam_iou")

    for i in [vcgpt_features, sam_iou, sam_hidden_states, sam_preds]:
        os.makedirs(i, exist_ok=True)

    # Initialize the CLIP model
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16)
    vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True).cuda()
    vision_tower.eval()

    # intialize sam model
    sam_image_processor = SamImageProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50", torch_dtype=torch.float16)
    sam_model = SamModel.from_pretrained(
        "Zigeng/SlimSAM-uniform-50", 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,).cuda()   

    sam_model.eval()

    all_videos = os.listdir(video_dir_path)

    x,y = 0,2500
    all_videos = all_videos[x:y]


    for video_name in tqdm(all_videos):
        video_path = f"{video_dir_path}/{video_name}"
        video_id = video_name.split('.')[0]
        
        try:
            video = load_video(video_path)
            
            vcgpts = []
            iou_scores = []
            preds = []
            sam_hids = []
            for i in range(len(video)):
                video_tensor = image_processor.preprocess(video[i], return_tensors='pt')['pixel_values']
                video_tensor = video_tensor.half().cuda()
                image_forward_outs = vision_tower(video_tensor, output_hidden_states=True)
                vcgpt_hidden_state = image_forward_outs.hidden_states[-2]  # torch.Size([1, 257, 1024])
                vcgpts.append(vcgpt_hidden_state)

                sam_tensor = sam_image_processor.preprocess(video[i], return_tensors="pt")['pixel_values'] 
                sam_tensor = sam_tensor.half().cuda() # (1,3,1024,1024)
                sam_forward_outs = sam_model(sam_tensor, output_hidden_states=True, return_dict=True)
                iou_score = sam_forward_outs.iou_scores #(1,1,3)
                pred_masks = sam_forward_outs.pred_masks  # torch.Size([1, 1, 3, 256, 256])
                sam_hidden_states = sam_forward_outs.vision_hidden_states[-1]   # torch.Size([1, 64, 64, 384])
                iou_scores.append(iou_score)
                preds.append(pred_masks)
                sam_hids.append(sam_hidden_states)

            stacked_vcpgts = torch.stack(vcgpts, dim=0)
            stacked_ious = torch.stack(iou_scores, dim=0)
            stacked_preds = torch.stack(preds, dim=0)
            stacked_sam_hids = torch.stack(sam_hids, dim=0)

            with open(f"{vcgpt_features}/{video_id}.pkl", 'wb') as f:
                    pickle.dump(stacked_vcpgts, f)
            with open(f"{sam_hidden_states}/{video_id}.pkl", 'wb') as f:
                pickle.dump(stacked_sam_hids, f)
            with open(f"{sam_preds}/{video_id}.pkl", 'wb') as f:
                pickle.dump(stacked_preds, f)
            with open(f"{sam_iou}/{video_id}.pkl", 'wb') as f:
                pickle.dump(stacked_ious, f)
    
        except Exception as e:
            print(f"Can't process {video_path}")

if __name__ == "__main__":
    main()  


# git pull; CUDA_VISIBLE_DEVICES=0 python scripts/save_sam_vcgpt_features.py --video_dir_path /data/shared/gauravs/llapsa/vcgpt_clips --clip_feat_path /data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos