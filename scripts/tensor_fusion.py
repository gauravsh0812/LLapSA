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
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim_f1, num_heads=8)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim_f1, num_heads=8)

    def forward(self, Fc, Fs):
        # Step 1: Self-Attention with layer normalization
        Fc_ln = self.layer_norm_fc(Fc)  # Layer normalization on Fc
        Fc_self_att, _ = self.self_attention(Fc_ln, Fc_ln, Fc_ln)  # Self-attention on Fc_ln
        Fc_1 = Fc_self_att + Fc  # Residual connection
        
        # Step 2: Cross-Attention with layer normalization
        Fc_1_ln = self.layer_norm_fc1(Fc_1)  # Layer normalization on Fc_1
        # print(Fc_1_ln.shape, Fs.shape)
        Fc_cross_att, _ = self.cross_attention(Fc_1_ln.transpose(0, 1), 
                                         Fs.transpose(0, 1), 
                                         Fs.transpose(0, 1))  # Cross-attention on Fc_1_ln and Fs
        # print("after CA: ", Fc_cross_att.shape, Fc_1.shape)
        Fc_2 = Fc_cross_att.permute(1,0,2) + Fc_1  # Residual connection

        return Fc_2

class TensorFusion(nn.Module):
    def __init__(self, root_path):
        super(TensorFusion,self).__init__()

        self.sam_hidden_states_path = os.path.join(root_path, "sam_hidden_states")
        self.vcgpt_features_path = os.path.join(root_path, "vcgpt_features")

        # Instantiate and use the attention module
        self.projection = nn.Linear(384, 1024)
        self.attention_module = AttentionModule(embed_dim_f1=1024)        
        self.lin_mat = nn.Linear(4096, 1024)
        self.final_lin = nn.Sequential(
                            nn.Linear(4096+256, 256*16),
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Linear(256*16, 256*8),
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Linear(256*8, 256),
                            nn.BatchNorm1d(1024),
                            nn.ReLU())

    def forward(self, video_name):
    # def forward(self, sam_hidden_states_tensor, vcgpt_features_tensor):
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
        
        sam_hidden_states_tensor = self.projection(sam_hidden_states_tensor)

        with open(vcgpt_features_pkl_path, 'rb') as file:
            vcgpt_features_tensor = pickle.load(file).cuda()

        vcgpt_features_tensor = vcgpt_features_tensor.squeeze(1)[:, 1:,:] # (100, 256, 1024)

        # print(sam_hidden_states_tensor.shape, vcgpt_features_tensor.shape)
        
        # cross attention on clip feature using sam features
        # it will have same shape as of vcgpt_features_tensor -- (100, 256, 1024)
        fc = self.attention_module(vcgpt_features_tensor, sam_hidden_states_tensor)        

        # cross attention on sam features using clip features
        # the shape will be == sam_hidden... shape -- (100, 4096, 1024)
        fs = self.attention_module(sam_hidden_states_tensor, vcgpt_features_tensor)

        # print(fs.shape)

        # element wise multiplication
        # Repeat tensor1 along the sequence length dimension to match tensor2
        fc_expanded = fc.repeat_interleave(16, dim=1)  # Shape becomes (100, 4096, 1024)
        elementwise_result = fc_expanded * fs  # torch.Size([100, 4096, 1024])
        # print("Element-wise multiplication result shape:", elementwise_result.shape)

        # bmm
        matrix_multiplication_result = torch.bmm(fc, fs.transpose(1, 2))  # Shape: (100, 256, 4096)
        # print("Matrix multiplication result shape:", matrix_multiplication_result.shape)

        # concatenate both multiplication results
        matrix_multiplication_result = self.lin_mat(matrix_multiplication_result) # (100, 256, 1024)
        mat_results = torch.cat((elementwise_result, matrix_multiplication_result),dim=1) # (100, 4096+256, 1024)

        # getting to the final format of (100, 256, 1024)
        final_vision_tensor = self.final_lin(mat_results.permute(0,2,1)).permute(0,2,1)
        # print(final_vision_tensor.shape)
        return final_vision_tensor


# sam = torch.rand((100,1,64,64,384))
# vcg = torch.rand((100,1,257,1024))
# tf = TensorFusion("data")
# tf(sam, vcg)