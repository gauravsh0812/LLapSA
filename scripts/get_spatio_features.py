import numpy as np
import os, pickle, tqdm

def get_spatio_temporal_features(features, num_temporal_tokens=100):
    t, s, c = features.shape
    
    temporal_tokens = np.mean(features, axis=1)
    padding_size = num_temporal_tokens - t
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')
    
    spatial_tokens = np.mean(features, axis=0)
    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)
    
    return sp_features

if __name__ == "__main__":
    select_hidden_state_layer = -2  # Selecting the hidden state layer
    
    L = os.listdir("/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/dino_features/")
    for i in tqdm.tqdm(L, total=len(L)):
        with open(f"/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/dino_features/{i}","rb") as f:
            image_forward_outs = pickle.load(f)

        select_hidden_state = image_forward_outs[select_hidden_state_layer]
        select_hidden_state = select_hidden_state[:, 1:].detach().cpu()  # Removing the CLS token
        select_hidden_state = get_spatio_temporal_features(select_hidden_state.numpy().astype("float16"))
        
        with open(f"/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/dino_sp_temp_features/{i}","wb") as f:
            pickle.dump(select_hidden_state, f)
