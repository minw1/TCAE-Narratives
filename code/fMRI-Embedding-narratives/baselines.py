import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class AE_Encoder(nn.Module):

    def __init__(self, d_vol=28549, d_emd=1024):
    
        super(AE_Encoder, self).__init__()
        self.en_emd = nn.Linear(d_vol, d_emd)
        
    def forward(self, x):
        return torch.tanh(self.en_emd(x))   


class AE_Decoder(nn.Module):

    def __init__(self, d_vol=28549, d_emd=1024):
    
        super(AE_Decoder, self).__init__()
        self.de_emd = nn.Linear(d_emd, d_vol)
        
    def forward(self, x):
        return torch.tanh(self.de_emd(x))  


class DSRAE_Encoder(nn.Module):

    def __init__(self, d_vol=28549, d_emd=128, d_lstm1=64, d_lstm2=32):
    
        super(DSRAE_Encoder, self).__init__()
        self.en_emd = nn.Linear(d_vol, d_emd)
        self.lstm1 = nn.LSTM(d_emd, d_lstm1, batch_first=True)
        self.lstm2 = nn.LSTM(d_lstm1, d_lstm2, batch_first=True)
        
    def forward(self, x):
    
        x = torch.tanh(self.en_emd(x))
        x = torch.tanh(self.lstm1(x)[0])
        return torch.tanh(self.lstm2(x)[0])   


class DSRAE_Decoder(nn.Module):

    def __init__(self, d_vol=28549, d_emd=128, d_lstm1=64, d_lstm2=32):
    
        super(DSRAE_Decoder, self).__init__()
        self.lstm1 = nn.LSTM(d_lstm2, d_lstm1, batch_first=True)
        self.lstm2 = nn.LSTM(d_lstm1, d_emd, batch_first=True)
        self.de_emd = nn.Linear(d_emd, d_vol)
        
    def forward(self, x):
        
        x = torch.tanh(self.lstm1(x)[0])
        x = torch.tanh(self.lstm2(x)[0])
        return torch.tanh(self.de_emd(x))   


class DVAE_Encoder(nn.Module):

    def __init__(self, d_vol=28549, d_emd1=256, d_emd2=128, d_emd3=64):
    
        super(DVAE_Encoder, self).__init__()
        self.en_emd1 = nn.Linear(d_vol, d_emd1)
        self.en_emd2 = nn.Linear(d_emd1, d_emd2)
        self.en_emd3 = nn.Linear(d_emd2, d_emd3)
        
    def forward(self, x):
    
        x = torch.tanh(self.en_emd1(x))
        x = torch.tanh(self.en_emd2(x))
        return torch.tanh(self.en_emd3(x))   


class DVAE_Decoder(nn.Module):

    def __init__(self, d_vol=28549, d_emd1=256, d_emd2=128, d_emd3=64):
    
        super(DVAE_Decoder, self).__init__()
        self.de_emd1 = nn.Linear(d_emd3, d_emd2)
        self.de_emd2 = nn.Linear(d_emd2, d_emd1)
        self.de_emd3 = nn.Linear(d_emd1, d_vol)
        
    def forward(self, x):
        
        x = torch.tanh(self.de_emd1(x))
        x = torch.tanh(self.de_emd2(x))
        return torch.tanh(self.de_emd3(x))
        

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    '''
    Compute 'Scaled Dot Product Attention'
    query: (nbatches, seq_len, d_k)
    key:   (nbatches, seq_len, d_k)
    value: (nbatches, seq_len, d_v) 
    
    For multi-head attention:
    query: (nbatches, h, seq_len, d_k)
    key:   (nbatches, h, seq_len, d_k)
    value: (nbatches, h, seq_len, d_k) 
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn        
        
        
class Self_Attention_Layer(nn.Module):
    def __init__(self, d_k, d_v):
        "Take in embedding size and number of heads."
        super(Self_Attention_Layer, self).__init__()
        # 3 for Q,K,V 
        self.linears = clones(nn.Linear(d_k, d_v), 3)
        self.attn = None
        
    def forward(self, query, key, value):
    
        nbatches = query.size(0)
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value)

        return x


class STAAE_Encoder(nn.Module):

    def __init__(self, d_vol=28549, d_emd1=512, d_emd2=128, d_latent=64):
    
        super(STAAE_Encoder, self).__init__()
        self.en_emd1 = nn.Linear(d_vol, d_emd1)
        self.en_emd2 = nn.Linear(d_emd1, d_emd2)
        self.attn = Self_Attention_Layer(d_emd2, d_latent)
        
    def forward(self, x):
    
        x = torch.tanh(self.en_emd1(x))
        x = torch.tanh(self.en_emd2(x))
        return self.attn(x,x,x)


class STAAE_Decoder(nn.Module):

    def __init__(self, d_vol=28549, d_emd1=512, d_emd2=128, d_latent=64):
    
        super(STAAE_Decoder, self).__init__()
        self.attn = Self_Attention_Layer(d_latent, d_emd2)
        self.de_emd1 = nn.Linear(d_emd2, d_emd1)
        self.de_emd2 = nn.Linear(d_emd1, d_vol)

    def forward(self, x):
        x = self.attn(x,x,x)
        x = torch.tanh(self.de_emd1(x))
        return torch.tanh(self.de_emd2(x))


class Standard_Autoencoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder):
        super(Standard_Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, vol):
        return self.decode(self.encode(vol))
    
    def encode(self, vol):
        return self.encoder(vol)
    
    def decode(self, latent):
        return self.decoder(latent)


class Variational_Autoencoder(nn.Module):
    """
    A standard Variational Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, d_emd, d_latent):
        super(Variational_Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.linear_mu = nn.Linear(d_emd, d_latent)
        self.linear_log_var = nn.Linear(d_emd, d_latent)
        
    def sample_z(self, mu, log_var):
        "Sample the random variable z as z = mu + std * epsilon"
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + std * eps
        
    def forward(self, vol):
        en = self.encoder(vol)
        mu = self.linear_mu(en)
        log_var = self.linear_log_var(en)
        z_latent = self.sample_z(mu, log_var)
        return self.decode(z_latent), mu, log_var
    
    def encode(self, vol):
        en = self.encoder(vol)
        mu = self.linear_mu(en)
        log_var = self.linear_log_var(en)
        return en, mu, log_var
    
    def decode(self, latent):
        return self.decoder(latent)

class vae_loss_function(nn.Module):
    def __init__(self):
        super(vae_loss_function, self).__init__()
        
    def forward(self, recon_x, x, mu, log_var):
        mse = F.mse_loss(recon_x,x,reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return mse + kld, mse
    

def Baseline_Autoencoder(model='DSRAE', d_vol=28549, d_emd1=512, d_emd2=64, d_latent=32):
    "Construct a model from hyperparameters."

    if model=='AE':
        model = Standard_Autoencoder(
                AE_Encoder(d_vol=d_vol, d_emd=d_latent),
                AE_Decoder(d_vol=d_vol, d_emd=d_latent)
                )

    elif model=='DSRAE':
        '''
        @inproceedings{li2019simultaneous,
        title={Simultaneous spatial-temporal decomposition of connectome-scale brain networks by deep sparse recurrent auto-encoders},
        author={Li, Qing and Dong, Qinglin and Ge, Fangfei and Qiang, Ning and Zhao, Yu and Wang, Han and Huang, Heng and Wu, Xia and Liu, Tianming},
        booktitle={International Conference on Information Processing in Medical Imaging},
        pages={579--591},
        year={2019},
        organization={Springer}
        }    
        '''
        model = Standard_Autoencoder(
                DSRAE_Encoder(d_vol=d_vol, d_emd=d_emd1, d_lstm1=d_emd2, d_lstm2=d_latent),
                DSRAE_Decoder(d_vol=d_vol, d_emd=d_emd1, d_lstm1=d_emd2, d_lstm2=d_latent)
                )

    elif model=='DVAE':
        '''
        @inproceedings{qiang2020deep,
        title={Deep Variational Autoencoder for Modeling Functional Brain Networks and ADHD Identification},
        author={Qiang, Ning and Dong, Qinglin and Sun, Yifei and Ge, Bao and Liu, Tianming},
        booktitle={2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI)},
        pages={554--557},
        year={2020},
        organization={IEEE}
        }
        '''
        model = Variational_Autoencoder(
                DVAE_Encoder(d_vol=d_vol, d_emd1=d_emd1, d_emd2=d_emd2, d_emd3=d_latent),
                DVAE_Decoder(d_vol=d_vol, d_emd1=d_emd1, d_emd2=d_emd2, d_emd3=d_latent), 
                d_emd=d_latent, 
                d_latent=d_latent
                )

    elif model=='DRVAE':
            '''
            @article{qiang2021modeling,
            title={Modeling and augmenting of fMRI data using deep recurrent variational auto-encoder},
            author={Qiang, Ning and Dong, Qinglin and Liang, Hongtao and Ge, Bao and Zhang, Shu and Sun, Yifei and Zhang, Cheng and Zhang, Wei and Gao, Jie and Liu, Tianming},
            journal={Journal of neural engineering},
            volume={18},
            number={4},
            pages={0460b6},
            year={2021},
            publisher={IOP Publishing}
            }
            '''
            model = Variational_Autoencoder(
                    DSRAE_Encoder(d_vol=d_vol, d_emd=d_emd1, d_lstm1=d_emd2, d_lstm2=d_latent),
                    DSRAE_Decoder(d_vol=d_vol, d_emd=d_emd1, d_lstm1=d_emd2, d_lstm2=d_latent),
                    d_emd=d_latent, 
                    d_latent=d_latent
                    )

    elif model=='STAAE':
            '''
            @inproceedings{dong2020spatiotemporal,
            title={Spatiotemporal Attention Autoencoder (STAAE) for ADHD Classification},
            author={Dong, Qinglin and Qiang, Ning and Lv, Jinglei and Li, Xiang and Liu, Tianming and Li, Quanzheng},
            booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
            pages={508--517},
            year={2020},
            organization={Springer}
            }
            '''
            model = Standard_Autoencoder(
                STAAE_Encoder(d_vol=d_vol, d_emd1=d_emd1, d_emd2=d_emd2, d_latent=d_latent),
                STAAE_Decoder(d_vol=d_vol, d_emd1=d_emd1, d_emd2=d_emd2, d_latent=d_latent)
                )
    else:
        raise NotImplementedError()


    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model



        
