import torch
from torch import nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d, n_tokens):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_encoding_matrix', self.generate_positional_encoding_matrix(d, n_tokens))
    def generate_positional_encoding_matrix(self, d, n_tokens):
        matrix = []
        for pos in range(n_tokens):
            matrix.append([pos/np.power(10000, 2*(i//2)/d) for i in range(d)])
        matrix = np.array(matrix)
        matrix[:, 0::2] = np.sin(matrix[:, 0::2])
        matrix[:, 1::2] = np.cos(matrix[:, 1::2])
        matrix = matrix[np.newaxis, ...]
        return torch.Tensor(matrix)
    def forward(self, tokens):
        return tokens + self.pos_encoding_matrix[:, :tokens.size(1)].detach().clone()

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, heads):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.heads = heads
        self.d_model = d_model
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, d_k*heads, bias=False)
        self.w_k = nn.Linear(d_model, d_k*heads, bias=False)
        self.w_v = nn.Linear(d_model, d_v*heads, bias=False)
        self.w_o = nn.Linear(heads * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
    def attention(self, q, k, v, mask=None):
        attn = torch.matmul(q / np.power(self.d_model, 0.5), k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask==0, np.inf)
        attn = nn.functional.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn
    def forward(self, q, k, v, mask=None):
        batch = q.size(0)
        t = q.size(1)
        residual = q 
        q = self.w_q(q).view(batch, t, self.heads, self.d_k)
        k = self.w_k(k).view(batch, t, self.heads, self.d_k)
        v = self.w_v(v).view(batch, t, self.heads, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.transpose(1,2).contiguous().view(batch, t, -1)
        output = self.w_o(output)
        output += residual
        output = self.norm(output)
        return output, attn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_inner):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        residual = x
        output = self.fc2(nn.functional.relu(self.fc1(x)))
        output += residual
        output = self.norm(output)
        return output

class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_k, d_v, heads):
        super(EncoderBlock, self).__init__()
        self.attn = MultiHeadAttention(d_model, d_k, d_v, heads)
        self.ffn = FeedForward(d_model, d_inner)
    def forward(self, x, attn_mask=None):
        output, attn_weights = self.attn(x, x, x, mask=attn_mask)
        output = self.ffn(output)
        return output, attn_weights

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_k, d_v, heads):
        super(DecoderBlock, self).__init__()
        self.masked_attn = MultiHeadAttention(d_model, d_k, d_v, heads)
        self.encoder_attn = MultiHeadAttention(d_model, d_k, d_v, heads)
        self.ffn = FeedForward(d_model, d_inner)
    def forward(self, decoder_input, encoder_output, attn_mask=None, encoder_attn_mask=None):
        decoder_output, masked_attn_weights = self.masked_attn(decoder_input, decoder_input, decoder_input, mask=attn_mask)
        decoder_output, encoder_attn_weights = self.encoder_attn(decoder_output, encoder_output, encoder_output, mask=encoder_attn_mask)
        decoder_output = self.ffn(decoder_output)
        return decoder_output, masked_attn_weights, encoder_attn_weights

class EncoderModel(nn.Module):
    def __init__(self, )
        super(EncoderModel, self).__init__()

