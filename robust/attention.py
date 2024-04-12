from typing import Any
import torch
from torch import nn as nn
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = (q / self.temperature) @ k.transpose(-2, -1) # swap the current -2 dim with -1 dim

        if mask is not None:
            attn = attn.masked_fill(~mask, -1e9)

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = attn @ v # output dim: [...,dim(k)[-2], dim(v)[-1]]

        return output, attn

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head # =5
        self.d_k = d_k # =7
        self.d_v = d_v # =8

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False) # (35, 35)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False) # (35, 35)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False) # (35, 40)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)   # (40, 35)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        q_org_shape = q.shape # [R, S, k, k, N, 35=3(RGB)+F(embedding)]
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head # 7, 8, 5

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(*q.shape[:-1], n_head, d_k) # [R, S, k, k, N, 5, 7]
        k = self.w_ks(k).view(*k.shape[:-1], n_head, d_k) # [R, S, k, k, N, 5, 7]
        v = self.w_vs(v).view(*v.shape[:-1], n_head, d_v) # [R, S, k, k, N, 5, 8]

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(-3, -2), k.transpose(-3, -2), v.transpose(-3, -2) # [R, S, k, k, 5, N, 7 or 8]

        if mask is not None:
            mask = mask.unsqueeze(-3)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask) # [R, S, k, k, 5, N, 8], [R, S, k, k, 5, N, N]

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(-3, -2).contiguous().view(*q_org_shape[:-1], -1) # [R, S, k, k, N, 40]
        # q = self.dropout(self.fc(q))
        q = self.fc(q) # [R, S, k, k, N, 35]
        q += residual

        q = self.layer_norm(q) 

        return q, attn # [R, S, k, k, N, 35], [R, S, k, k, 5, N, N]

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class PositionWiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        # x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


if __name__ == "__main__":
    mha = MultiHeadAttention()
    mha.apply()