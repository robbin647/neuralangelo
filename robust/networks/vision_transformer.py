
import torch.nn as nn
from typing import Dict
import sys
sys.path.insert(0, '/root/autodl-tmp/code/neuralangelo')
from robust.attention import MultiHeadAttention
from robust.configs.dotdict_class import DotDict

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
        self.encoder = ViTBlock(config)
        self.encoder_mask = None # TODO
    
    def forward(self, x):
        output, attn_weights = self.encoder(x, mask=self.encoder_mask)
        return output, attn_weights
    
    def compute_mask(self):
        return nn.Transformer.generate_square_subsequent_mask(self.config.embedding_size)

if __name__ == '__main__':
    import torch

    ViTConfig = DotDict({
        "num_encoder_block": 1,
        "n_head": 5,    
        "embedding_size": 35,
        "model_k_size": 7,
        "model_v_size": 7,
        "mlp_dim": 256, #  dimension of the intermediate output between last two linear layers 
        "dropout_rate": 0.1
    })

    oneViT = ViTModel(ViTConfig)
    input = torch.randn(512, 64, 1, 1, 8, 35)
    print(oneViT)
    out, attn_weights = oneViT(input)
    print(out.shape) # [512, 64, 1, 1, 8, 35]
    print(attn_weights.shape) # [512, 64, 1, 1, 5, 8, 8]