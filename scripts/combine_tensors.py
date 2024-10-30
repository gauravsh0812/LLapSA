import os, torch, pickle
import torch.nn as nn
from transformers import ViTModel


class CombineTensors(nn.Module):
    def __init__(self, root_path):
        super(CombineTensors, self).__init__()
        self.sam_hidden_states_path = os.path.join(root_path, "sam_hidden_states")
        self.sam_preds_path = os.path.join(root_path, "sam_preds")
        self.vcgpt_features_path = os.path.join(root_path, "vcgpt_features")

        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_model.eval()
        for param in self.vit_model.parameters():
            param.requires_grad = False  # Freeze all parameters

        # dealing with sam hidden states
        self.sam_hid_lin1 = nn.Sequential(
            nn.Linear(64*64, 1024), 
            nn.Linear(1024, 256)
        )
        self.sam_hid_lin2 = nn.Linear(384, 1024)

        # dealing with sam preds states 
        self.sam_pred_lin1 = nn.Linear(256, 256)
        self.sam_pred_lin2 = nn.Linear(256, 1024)

        # combined tensor 
        self.combined_tensor_lin = nn.Linear(256*2, 256)

    def split_tensor(self,tensor):
        # Split into 5 tensors of shape (20, x, y)
        split_tensors = torch.split(tensor, 20, dim=0)
        return split_tensors

    def forward(self, video_name):
        # pkl paths 
        sam_hidden_states_pkl_path = os.path.join(
            self.sam_hidden_states_path, f"{video_name}"
        )
        sam_preds_pkl_path = os.path.join(
            self.sam_preds_path, f"{video_name}"
        )
        vcgpt_features_pkl_path = os.path.join(
            self.vcgpt_features_path, f"{video_name}"
        )

        # loading the tensors
        with open(sam_hidden_states_pkl_path, 'rb') as file:
            sam_hidden_states_tensor = pickle.load(file).cuda()

        with open(sam_preds_pkl_path, 'rb') as file:
            sam_preds_tensor = pickle.load(file).cuda()

        with open(vcgpt_features_pkl_path, 'rb') as file:
            vcgpt_features_tensor = pickle.load(file).cuda()
        
        # modifying the preds
        squeezed_tensor = sam_preds_tensor.squeeze(1).squeeze(1)  # Now the shape is (100, 3, 256, 256)
        # Split the tensor along the channel axis (dim=1) -- (100, 256,256)
        mask1 = squeezed_tensor[:, 0, :, :].cuda()
        mask1 = self.sam_pred_lin1(mask1.permute(0,2,1)).permute(0,2,1)  # (100,257,256)
        mask1 = self.sam_pred_lin2(mask1)  # (100,257,1024)

        mask2 = squeezed_tensor[:, 1, :, :].cuda()
        mask2 = self.sam_pred_lin1(mask2.permute(0,2,1)).permute(0,2,1)  # (100,257,256)
        mask2 = self.sam_pred_lin2(mask2)  # (100,257,1024)
        
        mask3 = squeezed_tensor[:, 2, :, :].cuda()
        mask3 = self.sam_pred_lin1(mask3.permute(0,2,1)).permute(0,2,1)  # (100,257,256)
        mask3 = self.sam_pred_lin2(mask3)  # (100,257,1024)

        # modify the sam hidden states
        sam_hidden_states_tensor = sam_hidden_states_tensor.squeeze(1) # (100, 64, 64, 384)
        sam_hidden_states_tensor = torch.flatten(sam_hidden_states_tensor, 
                                                 start_dim=1, end_dim=2) # (100, 64*64, 384)
        
        sam_hidden_states_tensor = sam_hidden_states_tensor.permute(0,2,1)
        sam_hidden_states_tensor = self.sam_hid_lin1(sam_hidden_states_tensor).permute(0,2,1) # (100, 256, 384)
        sam_hidden_states_tensor = self.sam_hid_lin2(sam_hidden_states_tensor) # (100, 256, 1024)

        # concatenating
        vcgpt_features_tensor = vcgpt_features_tensor.squeeze(1)[:, 1:,:] # (100, 256, 1024)

        sam_tnsrs, vcgpt_tsnrs = self.split_tensor(sam_hidden_states_tensor), self.split_tensor(vcgpt_features_tensor)
        tnsrs = []
        for s,v in zip(sam_tnsrs, vcgpt_tsnrs):
            combined_tesnor = torch.cat((s,v),dim=1)  # (20, 256*2, 1024)
            combined_tesnor = self.combined_tensor_lin(combined_tesnor.permute(0,2,1)).permute(0,2,1)  # (20, 256, 1024)
            combined_outputs = self.vit_model(combined_tesnor)
            print("combvined_output sahpe: ", combined_outputs.shape)

            exit()
            tnsrs.append(combined_tesnor)
        exit()

        return combined_tesnor