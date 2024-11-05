import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

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
        Fc_cross_att, _ = self.cross_attention(Fc_1_ln.transpose(0, 1), 
                                         Fs.transpose(0, 1), 
                                         Fs.transpose(0, 1))  # Cross-attention on Fc_1_ln and Fs
        # print("after CA: ", Fc_cross_att.shape, Fc_1.shape)
        Fc_2 = Fc_cross_att.permute(1,0,2) + Fc_1  # Residual connection

        return Fc_2

class TensorFusion(nn.Module):
    def __init__(self, ):
        super(TensorFusion,self).__init__()

        # Instantiate and use the attention module
        self.projection1 = nn.Linear(384, 1024)
        self.projection2 = nn.Sequential(
            nn.Linear(64*64, 1024),
            nn.Linear(1024, 256),
        )
        self.attention_module = AttentionModule(embed_dim_f1=1024)        
        self.lin_mat = nn.Linear(256, 1024)
        self.final_lin = nn.Sequential(
                            nn.Linear(256+256, 256),
                            nn.BatchNorm1d(1024),
                            nn.ReLU())

    def get_spatio_temporal_features(self, features, num_temporal_tokens=100):
        # Ensure compatibility for both PyTorch tensors and NumPy arrays
        is_torch_tensor = isinstance(features, torch.Tensor)
        
        if is_torch_tensor:
            # PyTorch tensor shape unpacking
            t, s, c = features.shape
            # Calculate temporal tokens using PyTorch mean
            temporal_tokens = features.mean(dim=1)
        else:
            # Numpy array shape unpacking
            t, s, c = features.shape
            # Calculate temporal tokens using Numpy mean
            temporal_tokens = np.mean(features, axis=1)
        
        padding_size = num_temporal_tokens - t
        if padding_size > 0:
            if is_torch_tensor:
                # Use torch's pad method
                temporal_tokens = torch.nn.functional.pad(
                    temporal_tokens, (0, 0, 0, padding_size), mode="constant", value=0
                )
            else:
                # Use numpy pad for numpy array
                temporal_tokens = np.pad(
                    temporal_tokens, ((0, padding_size), (0, 0)), mode='constant'
                )
        
        if is_torch_tensor:
            spatial_tokens = features.mean(dim=0)
            sp_features = torch.cat([temporal_tokens, spatial_tokens], dim=0)
        else:
            spatial_tokens = np.mean(features, axis=0)
            sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)

        return sp_features

    def forward(self, video_features):

        sam_hidden_states_tensor, vcgpt_features_tensor = video_features 
        # print("shapes: ", sam_hidden_states_tensor.shape, vcgpt_features_tensor.shape)
        # shapes:  torch.Size([4, 100, 1, 64, 64, 384]) torch.Size([4, 100, 1, 257, 1024])

        sam_hidden_states_tensor = sam_hidden_states_tensor.squeeze(2) # (B, 100, 64, 64, 384)
        sam_hidden_states_tensor = torch.flatten(sam_hidden_states_tensor, 
                                                 start_dim=2, end_dim=3) # (B, 100, 64*64, 384)
        
        sam_hidden_states_tensor = self.projection1(sam_hidden_states_tensor)
        sam_hidden_states_tensor = self.projection2(sam_hidden_states_tensor.permute(0,1,3,2)).permute(0,1,3,2) # (B, 100, 256, 384)
        vcgpt_features_tensor = vcgpt_features_tensor.squeeze(2)[:,:,1:,:] # (B, 100, 256, 1024)

        # print(sam_hidden_states_tensor.shape, vcgpt_features_tensor.shape)

        
        final_vision_tensor = []
        for b in range(sam_hidden_states_tensor.shape[0]):
            # cross attention on clip feature using sam features
            # it will have same shape as of vcgpt_features_tensor -- (100, 256, 1024)
            temp_vcgpt_features_tensor = vcgpt_features_tensor[b,:,:,:]
            temp_sam_hidden_states_tensor = sam_hidden_states_tensor[b,:,:,:]

            fc = self.attention_module(temp_vcgpt_features_tensor, temp_sam_hidden_states_tensor)
            
            # cross attention on sam features using clip features
            # the shape will be == sam_hidden... shape -- (100, 256, 1024)
            fs = self.attention_module(temp_sam_hidden_states_tensor, temp_vcgpt_features_tensor)

            # print("fc, fs: ", fc.shape, fs.shape)
            # fc, fs:  torch.Size([100, 256, 1024]) torch.Size([100, 256, 1024])

            # element wise multiplication
            elementwise_result = fc * fs  # torch.Size([100, 256, 1024])
            # print("Element-wise multiplication result shape:", elementwise_result.shape)

            # bmm
            matrix_multiplication_result = torch.bmm(fc, fs.transpose(1, 2))  # Shape: (100, 256, 256)
            # print("Matrix multiplication result shape:", matrix_multiplication_result.shape)

            # concatenate both multiplication results
            matrix_multiplication_result = self.lin_mat(matrix_multiplication_result) # (100, 256, 1024)
            mat_results = torch.cat((elementwise_result, matrix_multiplication_result),dim=1) # (100, 256+256, 1024)

            # getting to the final format of (100, 256, 1024)
            final_tensor = self.final_lin(mat_results.permute(0,2,1)).permute(0,2,1)

            # spatial and temporal pooling
            final_tensor = self.get_spatio_temporal_features(final_tensor)
            final_vision_tensor.append(final_tensor)

        final_vision_tensor = torch.stack(final_vision_tensor, dim=0) # (B, 100+256, 1024)
        # print(final_vision_tensor.shape)
        return final_vision_tensor


# sam = torch.rand((100,1,64,64,384))
# vcg = torch.rand((100,1,257,1024))
# tf = TensorFusion("data")
# tf(sam, vcg)
