
import pdb
import torch.nn as nn
from typing import Dict
import sys
import math
sys.path.insert(0, '/root/autodl-tmp/code/neuralangelo')
from robust.attention import MultiHeadAttention
from robust.configs.dotdict_class import DotDict

class RTMLP(nn.Module): # MLP specifically defined for ray transformer
    def __init__(self, config):
        super().__init__()
        self.kernel_size = config.kernel_size 
        self.attn_embedding_reshape_dim = math.prod(config.kernel_size)*config.embedding_size
        self.fc1 = nn.Linear(self.attn_embedding_reshape_dim, config.final_out_size)
        self.activ_fn = nn.functional.gelu
        self.dropout = nn.Dropout(config.dropout_rate)

        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
    
    def forward(self, x):
        x = x.reshape(*x.shape[:3], -1, self.attn_embedding_reshape_dim) # [B, R, S, N_SRC, k*k*3]
        x = self.fc1(x)
        x = self.activ_fn(x)
        x = self.dropout(x)
        x = x.reshape(*x.shape[:3], -1) # [B, R, S, 256]
        return x

class RTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_size = config.embedding_size
        self.n_head = config.n_head
        assert self.embedding_size % self.n_head == 0, "Embedding size should be divisable by number of heads"
        self.model_k_size = self.embedding_size // self.n_head
        self.msa_layer = MultiHeadAttention(n_head=self.n_head, 
                                            d_model=config.embedding_size, 
                                            d_k=self.model_k_size, 
                                            d_v=config.model_v_size)
        # layer norm before self-attention
        self.ln_before = nn.LayerNorm(self.embedding_size, eps=1e-6)
        # layer norm after self-attention
        self.ln_after = nn.LayerNorm(self.embedding_size, eps=1e-6)
        # the final MLP layer
        self.mlp = RTMLP(config) 
    
    def forward(self, x, mask=None):
        x_1 = x
        x = self.ln_before(x)
        x, weights = self.msa_layer(x, x, x, mask) # [B,R,S,N_SRC*k*k,3]
        x = x + x_1
        x = self.ln_after(x)
        x = self.mlp(x) # [B,R,S,256]
        return x, weights


class ViTMLP(nn.Module):
    def __init__(self, config):
        super(ViTMLP, self).__init__()
        self.fc1 = nn.Linear(config.embedding_size, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.embedding_size)
        self.activ_fn = nn.functional.gelu
        self.dropout = nn.Dropout(config.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, config):
        super(ViTBlock, self).__init__()
        self.embedding_size = config.embedding_size
        self.n_head = config.n_head
        assert self.embedding_size % self.n_head == 0, "Embedding size should be divisable by number of heads"
        self.model_k_size = self.embedding_size // self.n_head
        self.msa_layer = MultiHeadAttention(n_head=self.n_head, 
                                            d_model=config.embedding_size, 
                                            d_k=self.model_k_size, 
                                            d_v=config.model_v_size)
        # layer norm before self-attention
        self.ln_before = nn.LayerNorm(self.embedding_size, eps=1e-6)
        # layer norm after self-attention
        self.ln_after = nn.LayerNorm(self.embedding_size, eps=1e-6)
        # the final MLP layer
        self.mlp = ViTMLP(config) 

    def forward(self, x, mask=None):
        x_1 = x
        x = self.ln_before(x)
        x, weights = self.msa_layer(x, x, x, mask)
        x = x + x_1
        x_2 = x
        x = self.ln_after(x)
        x = self.mlp(x)
        x = x + x_2
        return x, weights
     


class ViTModel(nn.Module):
    """
    Customized Vision Transformer without input embedding
    """
    def __init__(self, config: Dict) -> None:
        super(ViTModel, self).__init__()
        self.config = config
        if config.mlp_type == "ViTMLP":
            self.encoder = ViTBlock(config)
        else:
            self.encoder = RTBlock(config)
        self.encoder_mask = None # TODO
    
    def forward(self, x):
        output, attn_weights = self.encoder(x, mask=self.encoder_mask)
        return output, attn_weights
    
    def compute_mask(self):
        return nn.Transformer.generate_square_subsequent_mask(self.config.embedding_size)

class NanViTModel(ViTModel):
    """
    Customized case: the final MLP output dimension is different from embedding dimension
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        assert config.final_out_size % config.seq_length == 0, "Warning: the mlp final output size cannot be divided by N_SRC"
        self.reshape_mlp = nn.Linear(config.embedding_size, config.final_out_size // config.seq_length)
        
    
    def forward(self, x):
        x, attn_weights = super().forward(x)
        x = self.reshape_mlp(x) # [R, S, N_SRC, FINAL_OUT_SIZE / N_SRC]
        x = x.reshape(x.shape[:-2] + (-1,)) # [R, S, FINAL_OUT_SIZE]
        return x, attn_weights

if __name__ == '__main__':
    import torch

    ViTConfig = DotDict({
        "num_encoder_block": 1,
        "n_head": 3,    
        "embedding_size": math.prod((3,3))*3,
        "final_out_size": 256, # controls the output size at the end of the last MLP 
        "kernel_size": [3, 3], # (optional) only when using RTMLP, the kernel size in feature extraction step
        "seq_length": 4, #(required for nan) the N_SRC a.k.a. number of views in feature extraction step
        "model_k_size": 9,
        "model_v_size": 9,
        "mlp_dim": 256, #  dimension of the intermediate output between last two linear layers 
        "dropout_rate": 0.1,
        "mlp_type": "ViTMLP" # possible values: "RTMLP", "ViTMLP"
    })

    # oneViT = ViTModel(ViTConfig)
    oneViT = NanViTModel(ViTConfig)
    input = torch.randn(512, 64, 4, 27) # [R,S,N_SRC,k*k*3]
    print(oneViT)
    out, attn_weights = oneViT(input)
    print(out.shape) # expect: [512, 64, 256]
    print(attn_weights.shape) # [R, S, n_head, N_SRC, N_SRC]