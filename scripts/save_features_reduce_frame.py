import torch
torch.cuda.current_device()
import os, random, math
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import CLIPVisionModel, CLIPImageProcessor
from llapsa.model.merge import merge_tokens


def load_video(vis_path, num_frm=100):
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)
    total_num_frm = min(total_frame_num, num_frm)
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    # a, H, W, _ = img_array.shape
    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    
    if img_array.shape[0] != num_frm:
        img_array = torch.from_numpy(img_array).permute(1, 2, 3, 0).float()
        img_array = torch.nn.functional.interpolate(img_array, size=num_frm)
        img_array = img_array.permute(3, 0, 1, 2).to(torch.uint8).numpy()
    
    img_array = img_array.reshape((1, num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(num_frm):
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
    parser = argparse.ArgumentParser(description="Inference extracting features")
    parser.add_argument("--video_dir_path", required=True, help="Path to read the videos from.")
    parser.add_argument("--clip_feat_path_local", required=True, help="Output dir to save the local features.")
    # parser.add_argument("--clip_feat_path_memory", required=True, help="The output dir to save the memory features.")
    parser.add_argument("--xy", required=True,)
    args = parser.parse_args()

    return args

def get_spatio_temporal_features(features, num_temporal_tokens=100):
    t, s, c = features.shape

    temporal_tokens = np.mean(features, axis=1)
    padding_size = num_temporal_tokens - t
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')

    spatial_tokens = np.mean(features, axis=0)
    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)

    return sp_features

def reduce_similar_frames(visual_emb_frame):
    
    "https://github.com/Vision-CAIR/LongVU/blob/1ca42869fd456ecfef8acdc2aaa01e43864431e0/longvu/cambrian_arch.py#L1474"
    
    window_size = 5
    assert visual_emb_frame.shape[0] % window_size == 0, "num frames should be multiple of 5!"

    new_visual_emb_frames = []
    max_visual_len = visual_emb_frame.shape[1] * (visual_emb_frame.shape[0] * 0.4)  # keeping 60% frames

    for start_idx in range(0, len(visual_emb_frame), 5):
        end_idx = min(start_idx + window_size, len(visual_emb_frame))
        chunk_feature = visual_emb_frame[start_idx:end_idx]  # 5, HW, C
        if len(chunk_feature) == 1:
            new_visual_emb_frames.append(chunk_feature[0])
            continue

        sim = torch.nn.functional.cosine_similarity(
            chunk_feature[0]
            .unsqueeze(0)
            .repeat_interleave(len(chunk_feature[1:]), dim=0),
            chunk_feature[1:],
            dim=-1,
        )
        new_visual_emb_frame = torch.cat(
            [chunk_feature[0],chunk_feature[1:].flatten(0, 1)[sim.flatten(0, 1) < 0.7]],
            dim=0,
        )
        new_visual_emb_frames.append(new_visual_emb_frame)

    reduced_visual_len = sum([x.shape[0] for x in new_visual_emb_frames])
    
    if reduced_visual_len > max_visual_len:
        factor = (reduced_visual_len - max_visual_len) % len(new_visual_emb_frames)
        force_remove = math.ceil(
            (reduced_visual_len - max_visual_len - factor)
            / len(new_visual_emb_frames)
        )

        # force removal 
        for chunk_i in range(len(new_visual_emb_frames)):
            new_visual_emb_frames[chunk_i] = new_visual_emb_frames[chunk_i][:-force_remove]

        # extra removal -- factor
        for _ in range(int(factor)):
            chunk_i = random.randint(0, len(new_visual_emb_frames) - 1)
            new_visual_emb_frames[chunk_i] = new_visual_emb_frames[chunk_i][:-1]
        
        new_visual_emb_frames = torch.cat(new_visual_emb_frames, dim=0)
        new_visual_emb_frames = new_visual_emb_frames[:int(max_visual_len), :]
        
    else:
        # if the video is shorter, keep it intact
        # we would not extract key frames rather just take the 50% alternate frames
        # step = 3
        # new_visual_emb_frames = visual_emb_frame[::step, :, :]  # Slicing to get [50, :, :]
        # new_visual_emb_frames = new_visual_emb_frames.flatten(0,1)

        # 60% frames
        total_frames = visual_emb_frame.shape[0]
        target_frames = int(total_frames * 0.4)
        indices = torch.linspace(0, total_frames - 1, steps=target_frames).round().long()
        new_visual_emb_frames = visual_emb_frame[indices, :].flatten(0, 1)

    new_visual_emb_frames = new_visual_emb_frames.view(
                            int(new_visual_emb_frames.shape[0]/visual_emb_frame.shape[1]),
                            visual_emb_frame.shape[1],
                            new_visual_emb_frames.shape[-1]
                            )
    return new_visual_emb_frames

def main():
    args = parse_args()
    print("========Arguments=============")
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))   # str, arg_type
    print("==============================")
    video_dir_path = args.video_dir_path
    clip_feat_path_local = args.clip_feat_path_local
    os.makedirs(clip_feat_path_local, exist_ok=True)
    
    pretrained_path = "openai/clip-vit-large-patch14"

    # Initialize the CLIP model
    image_processor = CLIPImageProcessor.from_pretrained(pretrained_path, torch_dtype=torch.float16)
    vision_tower = CLIPVisionModel.from_pretrained(
                                            pretrained_path, 
                                            torch_dtype=torch.float16,
                                            low_cpu_mem_usage=True,
                                        )

    vision_tower.cuda()
    vision_tower.eval()
    for n, p in vision_tower.named_parameters():
        p.requires_grad_(False)
    
    x, y = args.xy.split("-")
    
    all_videos = [] 
    for i in os.listdir(video_dir_path):
        if "_60sec_" in i or "_45sec_part_" in i:
            all_videos.append(i)
    print(len(all_videos))
    all_videos = all_videos[int(x):int(y)]

    video_features = {}
    counter = 0
    for video_name in tqdm(all_videos):
        video_path = f"{video_dir_path}/{video_name}"
        video_id = video_name.split('.')[0]

        # if os.path.exists(f"{clip_feat_path_memory}/{video_id}.pkl") and os.path.exists(f"{clip_feat_path_local}/{video_id}.pkl"):  # Check if the file is already processed
        #     print(f"{video_id}.pkl exist")
        #     continue
        try:
            video = load_video(video_path)
            video_tensor = image_processor.preprocess(video, return_tensors='pt')['pixel_values']
            video_tensor = video_tensor.half().cuda()

            with torch.no_grad():
                image_forward_outs = vision_tower(video_tensor, output_hidden_states=True)

            if not os.path.exists(f"{clip_feat_path_local}/{video_id}.pkl"):
                last_state = image_forward_outs.hidden_states[-2][:, 1:]
                reduced_states = reduce_similar_frames(last_state)
                video_features[video_id] = get_spatio_temporal_features(reduced_states.half().cpu().numpy())
                counter += 1

        except Exception as e:
            print(f"Can't process {video_path}: {e}")

        if counter % 50 == 0:  # Save after every 50 videos, update this number as per your requirements
            for key in video_features.keys():
                clip_video_path = f"{clip_feat_path_local}/{key}.pkl"
                if not os.path.exists(clip_video_path):
                    features = video_features[key]
                    with open(clip_video_path, 'wb') as f:
                        pickle.dump(features, f)
                
            video_features = {}
        
    for key in video_features.keys():
        clip_video_path = f"{clip_feat_path_local}/{key}.pkl"
        if not os.path.exists(clip_video_path):
            features = video_features[key]
            with open(clip_video_path, 'wb') as f:
                pickle.dump(features, f)
    
    print("successfully processed {} videos, total video number: {}".format(counter, len(all_videos)))


if __name__ == "__main__":
    main()