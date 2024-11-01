import os, pickle, tqdm, multiprocessing

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

with multiprocessing.Pool(150) as pool:
    sam_hids = "/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/sam_hidden_states"
    pool.map(main, os.listdir(sam_hids))

