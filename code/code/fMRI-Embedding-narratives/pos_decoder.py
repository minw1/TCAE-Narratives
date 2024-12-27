import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Embeddings

class POS_Decoder(nn.Module):
    """
    Gives a prob. distribution over tokens given target(POS) and memory(fmri embedding) sequences.
    """
    def __init__(
        self, 
        l_vocab,
        d_mem=64,
        d_latent=64,
        d_ff=128,
        dropout=0.1,
        n_head=8,
        n_blocks=4
        ): 
        """
        Args:
            

        """
        super(POS_Decoder, self).__init__()
        
        self.embed = Embeddings(d_latent,l_vocab)
        self.pos_encoder = PositionalEncoding(d_latent, dropout, max_len=200)
        self.dec_layer = nn.TransformerDecoderLayer(d_model=d_mem, nhead=n_head, dim_ff=d_ff, dropout=dropout, batch_first=True)
        self.dec = nn.TransformerDecoder(self.dec_layer,n_blocks,norm=nn.LayerNorm(d_latent))
        self.token_predict = nn.Linear(d_latent, l_vocab)
        

    def forward(self, tgt_onehot, mem):
        x = self.embed(tgt_onehot)
        x = self.pos_encoder(x)
        x = self.dec(x,mem,tgt_is_causal=True)
        return self.token_predict(x)

class POS_predictor(nn.Module):
    def __init__(self, l_vocab, d_mem=64, d_dec_latent=64, d_dec_ff=128, dec_dropout=0.1, n_dec_head=8, n_dec_blocks=4, d_vol=86, d_enc_ff=128,n_enc_head=4,n_enc_blocks=4,enc_dropout=0.1):
        super(POS_predictor, self).__init__()
        self.autoencoder = Make_Autoencoder(N=n_enc_blocks, d_vol=d_vol, d_latent=d_mem, d_ff=d_enc_ff, h=n_enc_head, dropout=enc_dropout)
        self.decoder = POS_Decoder(l_vocab, d_mem=d_mem, d_latent=d_latent, d_ff=d_ff, dropout=dropout, n_head=n_head, n_blocks=n_blocks)
    def initialize_encoder_weights(self,checkpoint_file):
        self.encoder.load_state_dict(torch.load(checkpoint_file["model"]))
        return 1

    def forward(self, trs, tgt_onehot):
        mem = self.autoencoder.encoder(trs)
        return self.decoder(tgt_onehot, mem)
