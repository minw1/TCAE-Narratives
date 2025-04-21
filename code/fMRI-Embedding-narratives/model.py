import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    '''
    For single-head attention:
    query: (nbatches, seq_len, d_k)
    key:   (nbatches, seq_len, d_k)
    value: (nbatches, seq_len, d_v) 
    
    For multi-head attention:
    query: (nbatches, h, seq_len, d_k)
    key:   (nbatches, h, seq_len, d_k)
    value: (nbatches, h, seq_len, d_v) 
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn        
        
        
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_emd_in, d_emd_out, dropout=0.1):
        "Take in embedding size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        
        assert d_emd_in % h == 0
        assert d_emd_out % h == 0
        self.d_k = d_emd_in // h
        self.d_v = d_emd_out // h
        
        self.h = h
        # 3 for Q,K,V and 1 for heads concatenation
        self.linears = nn.ModuleList([nn.Linear(d_emd_in,d_emd_in),  nn.Linear(d_emd_in,d_emd_in),
                                      nn.Linear(d_emd_in,d_emd_out), nn.Linear(d_emd_out,d_emd_out)])

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        # Same mask applied to all h heads.
        if mask is not None:
            mask = mask.unsqueeze(1)
            
        nbatches = query.size(0)
        
        '''
        1) Do all the linear projections in batch
           input_size:  (nbatches, sql_len, d_emd_in)
           output size: query: (nbatches, h, seq_len, d_k)
                        key:   (nbatches, h, seq_len, d_k)
                        value: (nbatches, h, seq_len, d_v)
        '''
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        value = self.linears[2](value).view(nbatches, -1, self.h, self.d_v).transpose(1, 2)
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        '''
        3) "Concat" using a view and apply a final linear. 
           input_size:  (nbatches, h, seq_len, d_v)
           output size: (nbatches, sql_len, d_emd_out)
        '''
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_v)
        
        return self.linears[-1](x)

        
class PositionwiseFeedForward(nn.Module):
    "Implement Feed Forward Network."
    def __init__(self, d_emd_in, d_ff, d_emd_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_emd_in, d_ff)
        self.w_2 = nn.Linear(d_ff, d_emd_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
        
class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, d_emd, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_emd))
        self.b_2 = nn.Parameter(torch.zeros(d_emd))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, d_emd, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(d_emd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))     


class SelfAttentionLayer(nn.Module):
    "Attention Layer is made up of self-attn and feed forward"
    def __init__(self, d_emd, self_attn, feed_forward, dropout):
        super(SelfAttentionLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.d_emd = d_emd
        self.sublayer = clones(ResidualConnection(self.d_emd, dropout), 2)


    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward) 


class Embeddings(nn.Module):
    def __init__(self, d_emd, d_vol):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(d_vol, d_emd)
        self.d_emd = d_emd

    def forward(self, x):
        # print('********_embedding_************')
        # print(x.shape)
        # print(self.lut(x).shape)
        return self.lut(x) * math.sqrt(self.d_emd)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_emd, dropout, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_emd)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_emd, 2) *
                             -(math.log(10000.0) / d_emd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        #print('&&&&&&&_position_&&&&&&&')
        #print(x.shape)
        #print(self.pe.shape)
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        # print(x.shape)
        return self.dropout(x)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, d_vol, d_latent, pos_encoder, layer, N):
        super(Encoder, self).__init__()
        self.en_emd = nn.Linear(d_vol, d_latent)
        self.pos_encoder = pos_encoder
        self.layers = clones(layer, N)

        self.norm = LayerNorm(layer.d_emd)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        # print('********_encoder_*******')
        # print(x.shape)
        x = self.en_emd(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        # print(x.shape)
        return self.norm(x)     


class Decoder(nn.Module):
    "Core decoder is a stack of N+1 layers"
    def __init__(self, d_vol, d_latent, layer, N):
        super(Decoder, self).__init__()
        self.de_emd = nn.Linear(d_latent, d_vol)
        #self.pos_encoder = pos_encoder
        self.layers = clones(layer, N)

        self.norm = LayerNorm(layer.d_emd)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        # print('********_decoder_*******')
        # print(x.shape)
        #x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        # print(self.de_emd(x).shape)
        return self.de_emd(x)


class Standard_Autoencoder(nn.Module):

    #A standard Encoder-Decoder architecture. Base for this and many 
    #other models.

    def __init__(self, encoder, decoder):
        super(Standard_Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        
    def forward(self, vol, encode_mask=None, decode_mask=None):
        for a in (vol, encode_mask, decode_mask):
            if a is not None:
                print(a.shape)

        return self.decode(self.encode(vol, encode_mask), decode_mask)
    
    def encode(self, vol, encode_mask):
        return self.encoder(vol, encode_mask)
    
    def decode(self, latent, decode_mask):
        return self.decoder(latent, decode_mask)

        
def Make_Autoencoder(N=1, d_vol=28549, d_latent=64, d_ff=128, h=4, dropout=0.1):
    "Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_latent, d_latent)
    fedfwd = PositionwiseFeedForward(d_latent, d_ff, d_latent, dropout)
    position = PositionalEncoding(d_latent, dropout, max_len=2000)

    model = Standard_Autoencoder(
            Encoder(d_vol, d_latent, c(position), SelfAttentionLayer(d_latent, c(attn), c(fedfwd), dropout), N),
            Decoder(d_vol, d_latent, SelfAttentionLayer(d_latent, c(attn), c(fedfwd), dropout), N)
            )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model