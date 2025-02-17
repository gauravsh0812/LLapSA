import os
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from llapsa.model.merge import merge_tokens
from transformers import AutoImageProcessor, Dinov2Model, Dinov2Config
    

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

def get_spatio_temporal_features(features, num_temporal_tokens=100):
    t, s, c = features.shape
    
    temporal_tokens = np.mean(features, axis=1)
    padding_size = num_temporal_tokens - t
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')
    
    spatial_tokens = np.mean(features, axis=0)
    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)
    
    return sp_features

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
    parser.add_argument("--xy", required=True)
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

class DinoFeatureExtractor:
    def __init__(self, model_name="facebook/dinov2-giant", device="cuda:0"):
        """
        Initializes the DINOv2 model and image processor for feature extraction.
        Args:
            model_name (str): Pre-trained DINOv2 model to use.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        configuration = Dinov2Config.from_pretrained(model_name)
        # print(configuration)
        configuration.hidden_size = 1024
        configuration.num_attention_heads = 16
        self.processor = AutoImageProcessor.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model = Dinov2Model(configuration).to(self.device)
        self.model.eval()

    def extract_features(self, frames, layer_index=-2):
        """
        Extract features from a batch of video frames at a specific encoder layer.
        
        Args:
            frames (torch.Tensor): A batch of preprocessed video frames, shape (batch_size, channels, height, width).
            layer_index (int): Encoder layer index to extract features from.
        
        Returns:
            torch.Tensor: Extracted features from the specified encoder layer.
        """
        with torch.no_grad():
            # Forward pass through the model
            outputs = self.model(frames, output_hidden_states=True)
            # Extract features from the desired layer
            features = outputs.hidden_states[layer_index]
            global_feature = torch.cat(
                [mem[:, :1] for mem in outputs.hidden_states], 
                dim=1).mean(0).squeeze(0).detach().cpu().numpy().astype("float16")
        return features, global_feature

    def preprocess_frames(self, frames):
        """
        Preprocesses video frames for input to the DINOv2 model.
        
        Args:
            frames (list or np.ndarray): List of frames or NumPy array of shape (batch_size, height, width, channels).
        
        Returns:
            torch.Tensor: Preprocessed frames as a tensor, shape (batch_size, channels, height, width).
        """
        # Convert frames to PIL Images and preprocess
        inputs = self.processor(frames, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

def main():

    args = parse_args()
    video_dir_path = "/data/shared/gauravs/llapsa/surgical_tutor/video_clips"
    
    # Initialize the DinoV2 model    
    x, y = args.xy.split("-")
    
    all_videos = [] 
    for i in os.listdir(video_dir_path):
        if "_60sec_" in i or "_45sec_part_" in i:
            all_videos.append(i)
    print(len(all_videos))
    all_videos = all_videos[int(x):int(y)]
    
    dino = DinoFeatureExtractor()

    for video_name in tqdm(all_videos):
        video_path = f"{video_dir_path}/{video_name}"
        i = video_name.split('.')[0]

        try:
            frames = load_video(video_path)
            preprocessed_frames = dino.preprocess_frames(frames)
            features, global_feat = dino.extract_features(preprocessed_frames, layer_index=-2)
            sp_tmp = get_spatio_temporal_features(features[:, 1:].detach().cpu().numpy().astype("float16"))
            local_feat = merge_tokens(features[:, 1:],
                                  r_merge_list=[2880, 1440, 720, 360, 180, 90, 40]).detach().cpu().numpy().astype("float16")
            
            with open(f"/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/dino_sp_temp_features/{i}","wb") as f:
                pickle.dump(sp_tmp, f)
            with open(f"/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/local_features/{i}","wb") as f:
                pickle.dump(local_feat, f)
            with open(f"/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/global_features/{i}","wb") as f:
                pickle.dump(global_feat, f)



        except Exception as e:
            print(f"Can't process {video_path} due to {e}")
    
        
    


if __name__ == "__main__":
    main()  
