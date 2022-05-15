from torch import nn
from torch.nn import (MultiheadAttention, LayerNorm)
import numpy as np
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d, dropout, max_tokens=5000):
        super().__init__()
        self.register_buffer('pos_encoding_matrix', self.generate_positional_encoding_matrix(d, max_tokens))
        self.dropout = nn.Dropout(dropout)
    def generate_positional_encoding_matrix(self, d, max_tokens):
        matrix = []
        for pos in range(max_tokens):
            matrix.append([pos/np.power(10000, 2*(i//2)/d) for i in range(d)])
        matrix = np.array(matrix)
        matrix[:, 0::2] = np.sin(matrix[:, 0::2])
        matrix[:, 1::2] = np.cos(matrix[:, 1::2])
        matrix = matrix[np.newaxis, ...]
        return torch.Tensor(matrix)
    def forward(self, tokens):
        return self.dropout(tokens + self.pos_encoding_matrix[:, :tokens.size(1)])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
    def forward(self, x):
        residual = x
        output = self.fc2(nn.functional.relu(self.fc1(x)))
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_inner):
        super().__init__()
        self.attn = MultiheadAttention(d_model, heads)
        self.ffn = FeedForward(d_model, d_inner)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        output = x
        output = self.norm1(output + self.attn(x, x, x, attn_mask=attn_mask, 
                                    key_padding_mask=key_padding_mask)[0])
        output = self.norm2(output + self.ffn(output))
        return output

class Encoder(nn.Module):
    def __init__(self, d_model, layers, heads, d_inner):
        super().__init__()
        self.d_model = d_model
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, heads, d_inner) for i in range(layers)
        ])
    def forward(self, embedding, src_mask=None, src_key_pad_mask=None):
        for layer in self.encoder_layers:
            embedding = layer(embedding, src_mask, src_key_pad_mask)
        return embedding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_inner):
        super().__init__()
        self.masked_self_attn = MultiheadAttention(d_model, heads)
        self.encoder_decoder_attn = MultiheadAttention(d_model, heads)
        self.ffn = FeedForward(d_model, d_inner)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
    def forward(self, trg_seq, enc_seq, trg_mask=None, enc_mask=None, 
                trg_key_padding_mask=None, enc_key_padding_mask=None):
        output = trg_seq
        output = self.norm1(output + self.masked_self_attn(trg_seq, trg_seq, trg_seq, attn_mask=trg_mask, 
                                            key_padding_mask=trg_key_padding_mask)[0])
        output = self.norm2(output + self.encoder_decoder_attn(output, enc_seq, enc_seq, attn_mask=enc_mask, 
                                            key_padding_mask=enc_key_padding_mask)[0])
        output = self.norm3(output + self.ffn(output))
        return output

class Decoder(nn.Module):
    def __init__(self, d_model, layers, heads, d_inner):
        super().__init__()
        self.d_model = d_model
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, heads, d_inner) for i in range(layers)
        ])
    def forward(self, trg_seq, enc_seq, trg_mask=None, enc_mask=None, 
                trg_key_padding_mask=None, enc_key_padding_mask=None):
        output = trg_seq
        for layer in self.decoder_layers:
            output = layer(output, enc_seq, trg_mask, enc_mask, trg_key_padding_mask, enc_key_padding_mask)
        return output

class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, 
                    num_decoder_layers, heads,
                    d_model, src_vocab_size, tgt_vocab_size,
                    d_inner = 512, dropout = 0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, num_encoder_layers, heads, d_inner)
        self.decoder = Decoder(d_model, num_encoder_layers, heads, d_inner)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    def forward(self, src, trg, src_mask,
                tgt_mask, src_padding_mask,
                tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        memory = self.encoder(src_emb, src_mask, src_padding_mask)
        outs = self.decoder(tgt_emb, memory, trg_mask=tgt_mask,
                                        trg_key_padding_mask=tgt_padding_mask, 
                                        enc_key_padding_mask=memory_key_padding_mask)
        logits = self.fc_out(outs)
        return logits

    def encode(self, src, src_mask):
        output = self.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)
        return output

    def decode(self, tgt, memory, tgt_mask):
        output = self.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory, tgt_mask)
        return output

