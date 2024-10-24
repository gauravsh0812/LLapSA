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

def load_and_stack_hidden_states(temp, video_id, 
                                 counter, sam_hidden, 
                                 sam_preds, sam_iou):

    # Iterate over each video ID
    for tnsr in ["sam_hidden","sam_preds","sam_iou"]:
        hidden_states = []
        for i in range(counter):
            with open(os.path.join(temp, f"{tnsr}_{video_id}_{i}.pkl"), 'rb') as f:
                hidden_state = pickle.load(f)
                hidden_states.append(hidden_state)

        stacked_states = torch.stack(hidden_states, dim=0)
            
        if tnsr == "sam_hidden":
            output_file = os.path.join(sam_hidden, f"{video_id}.pkl")
        elif tnsr == "sam_preds":
            output_file = os.path.join(sam_preds, f"{video_id}.pkl")
        else:
            output_file = os.path.join(sam_iou, f"{video_id}.pkl")
        
        with open(output_file, 'wb') as f:
            pickle.dump(stacked_states, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--video_dir_path", required=True, help="Path to read the videos from.",
                        default="/data/shared/gauravs/llapsa/vcgpt_clips")
    parser.add_argument("--clip_feat_path", required=True, help="The output dir to save the features in.",
                        default="/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos")
    args = parser.parse_args()

    return args


def main():

    x,y = 9000, 12000
    n = 3

    args = parse_args()
    video_dir_path = args.video_dir_path
    clip_feat_path = args.clip_feat_path
    sam_preds = os.path.join(clip_feat_path,"sam_preds")
    sam_iou = os.path.join(clip_feat_path,"sam_iou")
    sam_hidden = os.path.join(clip_feat_path,"sam_hidden_states")
    temp = os.path.join(clip_feat_path, f"temp_{n}")
    for i in [clip_feat_path,sam_preds,sam_iou,sam_hidden, temp]:
        os.makedirs(i, exist_ok=True)

    sam_image_processor = SamImageProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50", torch_dtype=torch.float16)
    sam_model = SamModel.from_pretrained(
        "Zigeng/SlimSAM-uniform-50", 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,).cuda()   
    
    sam_model.eval()

    all_videos = os.listdir(video_dir_path)
    all_videos = all_videos[x:y]

    for video_name in tqdm(all_videos):
        video_path = f"{video_dir_path}/{video_name}"
        video_id = video_name.split('.')[0]
        try:
            video = load_video(video_path)
            counter = 0
        
            for i in range(len(video)):
                clip = video[i] #(224,224)
                sam_tensor = sam_image_processor.preprocess(clip, return_tensors="pt")['pixel_values']
                sam_tensor = sam_tensor.half().cuda() # (1,3,1024,1024)
                sam_forward_outs = sam_model(sam_tensor, output_hidden_states=True, return_dict=True)
                iou_score = sam_forward_outs.iou_scores # (1,1,3)
                pred_masks = sam_forward_outs.pred_masks  # torch.Size([1, 1, 3, 256, 256])
                sam_hidden_states = sam_forward_outs.vision_hidden_states[-1]   # torch.Size([1, 64, 64, 384])

                with open(f"{temp}/sam_hidden_{video_id}_{i}.pkl", 'wb') as f:
                    pickle.dump(sam_hidden_states, f)
                with open(f"{temp}/sam_preds_{video_id}_{i}.pkl", 'wb') as f:
                    pickle.dump(pred_masks, f)
                with open(f"{temp}/sam_iou_{video_id}_{i}.pkl", 'wb') as f:
                    pickle.dump(iou_score, f)
                
                counter +=1
            
            assert counter == len(video)
            load_and_stack_hidden_states(temp, video_id, counter, sam_hidden, sam_preds, sam_iou)

            # clear the temp
            for item in os.listdir(temp):
                item_path = os.path.join(temp, item)
                os.remove(item_path)
        
        except:
            print(f"Can't process {video_path}")

if __name__ == "__main__":
    main()
