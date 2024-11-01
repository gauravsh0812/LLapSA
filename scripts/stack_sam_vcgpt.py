import os, torch, pickle

sam_hids = "/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/sam_hidden_states"
vcgpt = "/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/vcgpt_features"

for s in os.listdir(sam_hids):

    with open(os.path.join(sam_hids,s), 'rb') as file:
        sam_tnsr = pickle.load(file).cuda()
    
    with open(os.path.join(vcgpt,s), 'rb') as file:
        vcgpt_tnsr = pickle.load(file).cuda()

    final = torch.stack((sam_tnsr, vcgpt_tnsr), dim=0)

    with open(f"/data/shared/gauravs/llapsa/sam_vcgpt_encoded_videos/stacked_sam_vcgpt/{s}.pkl", 'wb') as f:
        pickle.dump(final, f)    