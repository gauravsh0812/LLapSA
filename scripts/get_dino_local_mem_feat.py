import torch
torch.cuda.current_device()
import os
import pickle
import tqdm, numpy
from llapsa.model.merge import merge_tokens

def main():

    clip_feat_path_local = "/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/local_features"
    clip_feat_path_memory = "/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/global_features"

    os.makedirs(clip_feat_path_local, exist_ok=True)
    os.makedirs(clip_feat_path_memory, exist_ok=True)
    
    select_hidden_state_layer = -2  # Selecting the hidden state layer
    
    L = os.listdir("/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/dino_features/")
    for i in tqdm.tqdm(L, total=len(L)):
        with open(f"/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/dino_features/{i}","rb") as f:
            image_forward_outs = pickle.load(f)
        
        select_hidden_state_local = image_forward_outs[select_hidden_state_layer]
        select_hidden_state_local = select_hidden_state_local[:, 1:]  # Removing the CLS token
        local_feat = merge_tokens(select_hidden_state_local,
                                  r_merge_list=[2880, 1440, 720, 360, 180, 90, 40]).detach().cpu().numpy().astype("float16")

        mem_arrays = [mem[:,:1] for mem in image_forward_outs]
        global_feat = torch.cat(mem_arrays, 
                                dim=1).mean(0).squeeze(0).detach().cpu().numpy().astype("float16")
        
        with open(f"/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/local_features/{i}","wb") as f:
            pickle.dump(local_feat, f)
        with open(f"/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/global_features/{i}","wb") as f:
            pickle.dump(global_feat, f)


if __name__ == "__main__":
    main()