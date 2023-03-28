import math
import copy
import numpy as np

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(1, d_model, bias=False)
        self.d_model = d_model
 
    def forward(self, x):
        return self.lut(x.unsqueeze(-1)) * math.sqrt(self.d_model)
    

def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.float) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_postion_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.float)
    relative_postion_if_large = torch.min(
        relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
    return relative_buckets
    
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=100000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        position = np.arange(d_model, dtype=float)
        relative_position = torch.tensor(np.array([position - i for i in range(max_len)]), dtype=torch.float32)
        pe = _relative_position_bucket(relative_position, num_buckets=max_len)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x_size = x.shape
        x = x.view(x.size(0), -1, x.size(-1))
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        x = x.view(x_size)
        
        return self.dropout(x)
    

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_q = query.shape
    d_k = key.shape
    d_v = value.shape
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_q[-1])
    if mask is not None:
        _MASKING_VALUE= -1e9 if scores.dtype == torch.float32 else -1e4
        scores = scores.view(d_q[0], d_q[1], d_q[2], -1).transpose(-2, -1).view(d_q[0], d_q[1], d_q[3], d_k[3], d_q[2]).masked_fill(mask[:, :, :d_q[3]] == 0, _MASKING_VALUE)
    p_attn = scores.view(d_q[0], d_q[1], -1, d_q[2]).transpose(-2, -1).view(d_q[0], d_q[1], d_q[2], d_q[3], d_k[3]).softmax(dim=-1)
    if dropout is not None:
        p_attn = nn.Dropout(p=0.1)(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        channels = query.size(2)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, channels, self.h, self.d_k).transpose(1, 3)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 3)
            .contiguous()
            .view(nbatches, -1, channels, self.h * self.d_k)
        )

        del query
        del key
        del value
        return self.linears[-1](x)
    

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features, dtype=torch.float32))
        self.b_2 = nn.Parameter(torch.zeros(features, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, input_channel=12, output_channel=4):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.memory_decoder = nn.Linear(input_channel, output_channel) # [states to control]

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = self.memory_decoder(memory.transpose(2, 3)).transpose(2, 3)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):
        return self.proj(x)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_enc_mask, src_dec_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.generate(self.decode(self.encode(src, src_enc_mask), src_dec_mask, tgt, tgt_mask))

    def encode(self, src, mask):
        return self.encoder(self.src_embed(src), mask)

    def decode(self, memory, src_mask, tgt, mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, mask)
    
    def generate(self, x):
        self.generator.proj.weight = nn.Parameter(torch.linalg.pinv(self.tgt_embed[0].lut.weight))
        return self.generator(x).view(x.shape[:-1])


# def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
class Transformer(nn.Module):
    def __init__(self, configs):
        super(EncoderDecoder, self).__init__()
        c = copy.deepcopy
        self.attn = MultiHeadedAttention(configs.d_h, configs.d_model)
        self.ff = PositionwiseFeedForward(configs.d_model, configs.d_ff, configs.dropout)
        self.position = PositionalEncoding(configs.d_model, configs.dropout)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(configs.d_model, c(self.attn), c(self.ff), configs.dropout), configs.layer_num),
            Decoder(DecoderLayer(configs.d_model, c(self.attn), c(self.attn), c(self.ff), configs.dropout), configs.layer_num),
            nn.Sequential(Embeddings(configs.d_model), c(self.position)),
            nn.Sequential(Embeddings(configs.d_model), c(self.position)),
            Generator(configs.d_model),
        )
    
    def forward(self, src, tgt, src_enc_mask, src_dec_mask, tgt_mask):
        return self.model(src, tgt, src_enc_mask, src_dec_mask, tgt_mask)

    def encode(self, src, mask):
        return self.model.encoder(self.src_embed(src), mask)

    def decode(self, memory, src_mask, tgt, mask):
        return self.model.decoder(self.tgt_embed(tgt), memory, src_mask, mask)
    

def get_Transformer(configs):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(configs.d_h, configs.d_model)
    ff = PositionwiseFeedForward(configs.d_model, configs.d_ff, configs.dropout)
    position = PositionalEncoding(configs.d_model, configs.dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(configs.d_model, c(attn), c(ff), configs.dropout), configs.layer_num),
        Decoder(DecoderLayer(configs.d_model, c(attn), c(attn), c(ff), configs.dropout), configs.layer_num),
        nn.Sequential(Embeddings(configs.d_model), c(position)),
        nn.Sequential(Embeddings(configs.d_model), c(position)),
        Generator(configs.d_model),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
    

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
    
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size, 4)
    subsequent_mask = torch.zeros(attn_shape, dtype=torch.float32)
    for i in range(size):
        for j in range(i+1, size):
            subsequent_mask[0][i][j]=1
    return subsequent_mask == 0