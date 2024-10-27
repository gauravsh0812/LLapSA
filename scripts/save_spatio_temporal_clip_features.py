import os
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from scripts.combine_tensors import CombineTensors

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

    parser.add_argument("--root_path", required=True, help="Path to read the videos from.")
    parser.add_argument("--clip_feat_path", required=True, help="The output dir to save the features in.")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    root_path = args.root_path
    all_videos = os.listdir(os.path.join(root_path, "sam_hidden_states"))[9000:]
    clip_feat_path = args.clip_feat_path
    os.makedirs(clip_feat_path, exist_ok=True)

    combine_tensors = CombineTensors(root_path).cuda().to(torch.float16)

    video_features = {}
    counter = 0
    for video_name in tqdm(all_videos):
        video_id = video_name.split(".")[0] 
        if os.path.exists(f"{clip_feat_path}/{video_id}"):  # Check if the file is already processed
            continue
        
        combine_features = combine_tensors(video_name)
        combine_features = combine_features.detach().cpu().numpy().astype("float16")
            
        video_features[video_id] = get_spatio_temporal_features(combine_features)
        counter += 1
        
        if counter % 512 == 0:  # Save after every 512 videos, update this number as per your requirements
            for key in video_features.keys():
                features = video_features[key]
                with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
                    pickle.dump(features, f)
            video_features = {}

    for key in video_features.keys():
        features = video_features[key]
        with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
            pickle.dump(features, f)


if __name__ == "__main__":
    main()