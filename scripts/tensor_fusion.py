import torch 
import os 
import pickle
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, embed_dim_f1):
        super(AttentionModule, self).__init__()
        # Define the layer normalization layers
        self.layer_norm_fc = nn.LayerNorm(embed_dim_f1)
        self.layer_norm_fc1 = nn.LayerNorm(embed_dim_f1)
        
        # Define the self-attention and cross-attention layers
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim_f1, num_heads=8)

    def forward(self, Fc, Fs):
        # Step 1: Self-Attention with layer normalization
        Fc_ln = self.layer_norm_fc(Fc)  # Layer normalization on Fc
        Fc_self_att, _ = self.attention(Fc_ln, Fc_ln, Fc_ln)  # Self-attention on Fc_ln
        Fc_1 = Fc_self_att + Fc  # Residual connection
        
        # Step 2: Cross-Attention with layer normalization
        Fc_1_ln = self.layer_norm_fc1(Fc_1)  # Layer normalization on Fc_1
        Fc_cross_att, _ = self.attention(Fc_1_ln, Fs, Fs)  # Cross-attention on Fc_1_ln and Fs
        Fc_2 = Fc_cross_att + Fc_1  # Residual connection

        return Fc_2

class TensorFusion(nn.Module):
    def __init__(self, root_path):
        super(TensorFusion,self).__init__()

        self.sam_hidden_states_path = os.path.join(root_path, "sam_hidden_states")
        self.vcgpt_features_path = os.path.join(root_path, "vcgpt_features")

        # Instantiate and use the attention module
        self.attention_module = AttentionModule(embed_dim_f1=1024)

    def forward(self, video_name):
        # pkl paths 
        sam_hidden_states_pkl_path = os.path.join(
            self.sam_hidden_states_path, f"{video_name}"
        )
        vcgpt_features_pkl_path = os.path.join(
            self.vcgpt_features_path, f"{video_name}"
        )

        # loading the tensors
        with open(sam_hidden_states_pkl_path, 'rb') as file:
            sam_hidden_states_tensor = pickle.load(file).cuda()
        
        sam_hidden_states_tensor = sam_hidden_states_tensor.squeeze(1) # (100, 64, 64, 384)
        sam_hidden_states_tensor = torch.flatten(sam_hidden_states_tensor, 
                                                 start_dim=1, end_dim=2) # (100, 64*64, 384)
        
        with open(vcgpt_features_pkl_path, 'rb') as file:
            vcgpt_features_tensor = pickle.load(file).cuda()

        vcgpt_features_tensor = vcgpt_features_tensor.squeeze(1)[:, 1:,:] # (100, 256, 1024)

        # cross attention on clip feature using sam features
        fc = self.attention_module(vcgpt_features_tensor, sam_hidden_states_tensor)

        # cross attention on sam features using clip features
        fs = self.attention_module(sam_hidden_states_tensor, vcgpt_features_tensor)
        