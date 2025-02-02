import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Embeddings, Make_Autoencoder, PositionalEncoding, MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, Encoder, SelfAttentionLayer

class POS_Decoder(nn.Module):
    """
    Gives a prob. distribution over tokens given target(POS) and memory(fmri embedding) sequences.
    """
    def __init__(
        self, 
        l_vocab,
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
        self.dec_layer = nn.TransformerDecoderLayer(d_model=d_latent, nhead=n_head, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.dec = nn.TransformerDecoder(self.dec_layer,n_blocks,norm=nn.LayerNorm(d_latent))
        self.token_predict = nn.Linear(d_latent, l_vocab)


    def forward(self, tgt_onehot, mem):
        x = self.embed(tgt_onehot)
        x = self.pos_encoder(x)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_onehot.shape[1])
        x = self.dec(x,mem,tgt_mask = tgt_mask, tgt_is_causal=True)
        x = self.token_predict(x)
        return x

class POS_predictor(nn.Module):
    def __init__(self, l_vocab, d_latent=64, d_dec_ff=128, dec_dropout=0.1, n_dec_head=8, n_dec_blocks=4, d_vol=86, d_enc_ff=128,n_enc_head=4,n_enc_blocks=4,enc_dropout=0.1):
        super(POS_predictor, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_enc_head, d_latent, d_latent)
        fedfwd = PositionwiseFeedForward(d_latent, d_enc_ff, d_latent, enc_dropout)
        position = PositionalEncoding(d_latent, enc_dropout, max_len=2000)
        self.encoder = Encoder(d_vol, d_latent, c(position), SelfAttentionLayer(d_latent, c(attn), c(fedfwd), enc_dropout), n_enc_blocks)
        self.decoder = POS_Decoder(l_vocab, d_latent=d_latent, d_ff=d_dec_ff, dropout=dec_dropout, n_head=n_dec_head, n_blocks=n_dec_blocks)

    def initialize_encoder_weights(self,checkpoint_file):
        state_dict = torch.load(checkpoint_file)['model']
        state_dict = {k.replace("module.encoder.", ""): v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if not "decoder" in k}
        self.encoder.load_state_dict(state_dict)
        return 1

    def forward(self, trs, pos_tokens=None):
        if pos_tokens is None:
            raise Exception("No pos_tokens")
        #print(f"trs shape: {trs.shape}")
        #print(f"pos_tokens shape: {pos_tokens.shape}")
        mem = self.encoder(trs, None)
        return self.decoder(pos_tokens, mem)


class No_Scans_POS_Decoder(nn.Module):
    """
    Gives a prob. distribution over tokens given target(POS) and memory(fmri embedding) sequences.
    """
    def __init__(
        self, 
        l_vocab,
        d_latent=64,
        d_ff=128,
        dropout=0.1,
        n_head=8,
        n_blocks=4
        ): 
        """
        Args:
            

        """
        super(No_Scans_POS_Decoder, self).__init__()
        self.embed = Embeddings(d_latent,l_vocab)
        self.pos_encoder = PositionalEncoding(d_latent, dropout, max_len=200)
        # A decoder-only model can be implemented with the pytorch "encoder," as it allows causal masks, and omits cross attention.
        self.dec_layer = nn.TransformerEncoderLayer(d_model=d_latent, nhead=n_head, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.dec = nn.TransformerEncoder(self.dec_layer,n_blocks,norm=nn.LayerNorm(d_latent))
        self.token_predict = nn.Linear(d_latent, l_vocab)

    def forward(self, tgt_onehot):
        x = self.embed(tgt_onehot)
        x = self.pos_encoder(x)
        x = self.dec(x,is_causal=True)
        x = self.token_predict(x)
        return x