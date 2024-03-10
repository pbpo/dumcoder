import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TernaryEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(TernaryEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, input):
        weight_ternary = self.ternary_weight()
        return nn.functional.embedding(input, weight_ternary)

    def ternary_weight(self):
        abs_mean = self.weight.abs().mean()
        mask = self.weight.abs() > abs_mean
        weight_ternary = torch.where(mask, self.weight.sign(), torch.zeros_like(self.weight))
        return weight_ternary
class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(TernaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def ternary_weight(self):
        abs_mean = self.weight.abs().mean()
        mask = self.weight.abs() > abs_mean
        weight_ternary = torch.where(mask, self.weight.sign(), torch.zeros_like(self.weight))
        return weight_ternary

    def forward(self, input):
        weight_ternary = self.ternary_weight()
        return F.linear(input, weight_ternary)

class TernaryMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(TernaryMultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_proj = TernaryLinear(d_model, d_model)
        self.key_proj = TernaryLinear(d_model, d_model)
        self.value_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)

        
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)

        
        attn_output = torch.matmul(attn_probs, value).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_output = self.out_proj(attn_output)

        return attn_output
class TernaryFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(TernaryFeedForward, self).__init__()
        self.linear1 = TernaryLinear(d_model, d_ff)
        self.linear2 = TernaryLinear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
class TernaryTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TernaryTransformerEncoderLayer, self).__init__()
        self.self_attn = TernaryMultiheadAttention(d_model, num_heads)
        self.feed_forward = TernaryFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x

class TernaryTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TernaryTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TernaryTransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TernaryTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TernaryTransformerDecoderLayer, self).__init__()
        self.self_attn = TernaryMultiheadAttention(d_model, num_heads)
        self.enc_dec_attn = TernaryMultiheadAttention(d_model, num_heads)
        self.feed_forward = TernaryFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_attn_mask=None, enc_dec_attn_mask=None):
        self_attn_output = self.self_attn(x, x, x, attn_mask=self_attn_mask)
        x = x + self.dropout1(self_attn_output)
        x = self.norm1(x)

        enc_dec_attn_output = self.enc_dec_attn(x, enc_output, enc_output, attn_mask=enc_dec_attn_mask)
        x = x + self.dropout2(enc_dec_attn_output)
        x = self.norm2(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)

        return x

class TernaryTransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TernaryTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TernaryTransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_output, self_attn_mask=None, enc_dec_attn_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, self_attn_mask, enc_dec_attn_mask)
        return x
class TernaryTransformer(nn.Module):
    def __init__(self, num_enc_layers, num_dec_layers, d_model, num_heads, d_ff, input_vocab_size, output_vocab_size, max_seq_len, dropout=0.1):
        super(TernaryTransformer, self).__init__()
        self.encoder = TernaryTransformerEncoder(num_enc_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TernaryTransformerDecoder(num_dec_layers, d_model, num_heads, d_ff, dropout)
        self.enc_embedding = TernaryEmbedding(input_vocab_size, d_model)
        self.dec_embedding = TernaryEmbedding(output_vocab_size, d_model)
        self.position_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        self.fc = TernaryLinear(d_model, output_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, enc_dec_attn_mask=None):
        src_embedded = self.enc_embedding(src)
        src_embedded = self.position_enc(src_embedded)
        enc_output = self.encoder(src_embedded, src_mask)

        tgt_embedded = self.dec_embedding(tgt)
        tgt_embedded = self.position_enc(tgt_embedded)
        dec_output = self.decoder(tgt_embedded, enc_output, tgt_mask, enc_dec_attn_mask)

        output = self.fc(dec_output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)