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
    L = os.listdir("/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/dino_features/")
    for i in tqdm.tqdm(L, total=len(L)):
        with open(f"/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/dino_features/{i}","rb") as f:
            tnsr = pickle.load(f)
        print(tnsr.shape)
        exit()
        sp_tmp = get_spatio_temporal_features(tnsr)
        with open(f"/data/shared/gauravs/llapsa/surgical_tutor/llapsa/dino/dino_sp_temp_features/{i}","wb") as f:
            pickle.dump(sp_tmp, f)