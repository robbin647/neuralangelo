import torch
import torch.nn as nn
from robust.attention import MultiHeadAttention
import pdb

input_channel = 35

class RayTransformer(nn.Module):
    def __init__(self,  d_model=35, output_dim=257, nhead=5, input_seq_len=9, output_seq_len=1):
        """
        input_seq_len (int): set to be k*k = 9
        output_seq_len (int): set to be 1
        """
        super(RayTransformer, self).__init__()
        # self.encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, num_layers=num_encoder_layers)
        self.mha_layer = MultiHeadAttention(5, input_channel, 7, 7)
        self.decoder = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        # self.embedding = nn.Embedding(input_dim, d_model)
        if d_model != output_dim:
            self.fc_out = nn.Linear(d_model, output_dim)
        else:
            self.fc_out = None
        self.d_model = d_model
        self.num_valid_obs = None #TODO: Handle this from NanMLP::compute_extended_features
        self.gamma_x = None #TODO: Handle this from NanMLP::compute_extended_features
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

    def aggr_feat_in(self, rgb_feat):
        """
        Input: rgb_feat [R, S, k, k, N, f+3=35]
        Output: [R, S, k, k, N, 36] (=rgb_feat+γ(x))
        """
        assert rgb_feat.size(-1) == 35
        pdb.set_trace()
        feat_weight, _ = self.mha_layer(rgb_feat, rgb_feat, rgb_feat, mask=self.num_valid_obs) # [R, S, k, k, N, 35]
        
        gamma_x = torch.randn(1, feat_weight.size()[-1]).expand(*feat_weight) if self.gamma_x is None else self.gamma_x
        # add spatial embedding γ(x)
        # feat_weight = feat_weight + gamma_x
        return feat_weight

    def forward(self, src, tgt=None):
        """
        INPUT
        src: [..., input_seq_len=9, 35]

        OUPUT
        output: [...,1,output_dim=257]
        """
        assert src.dim() >= 3, "`src` must be at least 3-dimensional (<batch>,<n_seq>,<n_embed>)"
        assert src.size()[-1] == self.d_model, "`src` feature embedding dimension does not match `d_model`"
        memory = self.aggr_feat_in(src)
        batch_size = src.size()[:-2]
        if tgt is None:
            tgt = torch.zeros(*batch_size, self.output_seq_len, self.d_model).to(src.device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.output_seq_len)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        if self.fc_out is not None:
            output = self.fc_out(output)
        return output

"""
    # Example usage:
    batch_size = 10
    input_length = 36
    input_dim = 100  # Vocabulary size
    output_dim = 257  # Output vector size
    d_model = 64
    nhead = 4
    num_encoder_layers = 1
    num_decoder_layers = 1

    # Create input and target tensors (random for demonstration)
    input_tokens = torch.randint(0, input_dim, (batch_size, input_length))
    target_tokens = torch.randn(batch_size, 1, d_model)  # Example output tokens

    model = Transformer(input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers)
    output = model(input_tokens, target_tokens)
    print("Output shape:", output.shape)  # Should be (batch_size, 1, output_dim)
"""