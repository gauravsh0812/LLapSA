import tqdm
import os, pickle

sam_hids = "/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/sam_hidden_states"
vcgpt = "/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/vcgpt_features"

def main(s):    

    with open(os.path.join(sam_hids,s), 'rb') as file:
        sam_tnsr = pickle.load(file)
    
    with open(os.path.join(vcgpt,s), 'rb') as file:
        vcgpt_tnsr = pickle.load(file)

    final = (sam_tnsr, vcgpt_tnsr)

    with open(f"/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/stacked_sam_vcgpt/{s}.pkl", 'wb') as f:
        pickle.dump(final, f)    

filenames = os.listdir("/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/sam_hidden_states")[:3]
for s in tqdm.tqdm(filenames, total=len(filenames)):
    print(s)
    main(s)