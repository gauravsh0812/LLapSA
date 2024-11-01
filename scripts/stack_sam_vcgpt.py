from tqdm import tqdm
import os, pickle
import multiprocessing

def main(s):
    sam_hids = "/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/sam_hidden_states"
    vcgpt = "/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/vcgpt_features"

    with open(os.path.join(sam_hids,s), 'rb') as file:
        sam_tnsr = pickle.load(file).cuda()
    
    with open(os.path.join(vcgpt,s), 'rb') as file:
        vcgpt_tnsr = pickle.load(file).cuda()

    final = (sam_tnsr, vcgpt_tnsr)

    with open(f"/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/stacked_sam_vcgpt/{s}.pkl", 'wb') as f:
        pickle.dump(final, f)    

filenames = os.listdir("/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/sam_hidden_states")
with multiprocessing.Pool(processes=150) as pool:
    results = list(tqdm(pool.imap(main, filenames), total=len(filenames)))
