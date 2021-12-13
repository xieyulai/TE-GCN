import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange

class Multi_Head_Temporal_Attention(nn.Module):
    def __init__(self,channels,H,T,A,inherent=1,norm='bn',dropout=0.1,with_cls_token=0,pe=1):
        super(Multi_Head_Temporal_Attention,self).__init__()
        self.n_joint = A.shape[-1]
        self.multi_head_attention = nn.ModuleList()
        self.head_num = H
        self.norm_type = norm
        self.drop = nn.Dropout(dropout)
        self.emb_dim = self.n_joint * channels

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(channels)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(channels*self.n_joint, channels*self.n_joint)

        self.residual = lambda x: x
        self.ffn = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        #self.ffn1 = nn.Conv2d(in_channels=channels, out_channels=2048, kernel_size=1)
        #self.ffn2= nn.Conv2d(in_channels=2048, out_channels=channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

        inter_channels = channels//H
        for i in range(H):
            self.multi_head_attention.append(Temporal_Attention(channels,inter_channels,T,A,inherent))

        if with_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, self.emb_dim)
            )
        else:
            self.cls_token = None

        #init = 'xavier'
        #if init == 'xavier':
            #self.apply(self._init_weights_xavier)
        #else:
            #self.apply(self._init_weights_trunc_normal)

        self.pe = pe
        if with_cls_token:
            PE_LEN = T+1
        else:
            PE_LEN = T

        if self.pe:
            self.pos_embedding = nn.Parameter(torch.randn(1,  PE_LEN, self.emb_dim))

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x):

        N,D,T,V = x.shape

        x = rearrange(x, 'n d t v -> n t (d v)')

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(N, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            T+=1

        if self.pe:
            x += self.pos_embedding[:, :T]

        x = rearrange(x, 'n t (d v) -> n d t v ',d=D,v=V)

        output = []
        for i in range(self.head_num):
            out = self.multi_head_attention[i](x)
            output.append(out)

        z = torch.cat(output, 2)  # (6,256,50,50)

        # norm + add

        if self.norm_type == 'ln':
            z = self.norm(z)

        z = rearrange(z, 'n t (d v) -> n d t v ',d=D,v=V)
        #z = z.reshape(N,T,-1,V).permute(0,2,1,3).contiguous()

        if self.norm_type == 'bn':
            z = self.norm(z)

        z += self.residual(x)
        z = self.drop(z)

        # ffn + norm + add
        z = self.relu(self.ffn(z)) + self.residual(x)
        #z = self.ffn2(self.relu(self.ffn1(z))) + self.residual(x)

        if self.norm_type == 'bn':
            z = self.norm(z)

        #z = z.permute(0,2,1,3).contiguous().reshape(N,T,-1)
        z = rearrange(z, 'n d t v -> n t (d v)')

        if self.norm_type == 'ln':
            z = self.norm(z)

        if self.cls_token is not None:
            cls_tokens = z[:, 0:1]
            z = z[:, 1:]

        # reshape
        #z = z.reshape(N,T,-1,V).permute(0,2,1,3).contiguous()
        z = rearrange(z, 'n t (d v) -> n d t v ',d=D,v=V)

        return z,cls_tokens

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    #return torch.from_numpy(subsequent_mask) == 0
    return torch.from_numpy(subsequent_mask) == 1

class Temporal_Attention(nn.Module):

    def __init__(self,in_channels,out_channels,T,A,inherent,is_pe=0):
        super(Temporal_Attention,self).__init__()
        self.n_joint = A.shape[-1]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_Q = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.embedding_K = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.embedding_V = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.attention = Attention()

        self.is_pe = is_pe
        if self.is_pe:
            self.pe = LocPositionalEncoder(out_channels*self.n_joint,0.0,300)

        self.inherent = inherent
        if self.inherent:
            self.PA = nn.Parameter(torch.eye(T))
            #self.PA = torch.eye(T,requires_grad=False)
            #self.alpha = nn.Parameter(torch.zeros(1))
            self.alpha = nn.Parameter(torch.ones(1))

    def forward(self,x):

        N,_,T,_ = x.shape

        Q = self.embedding_Q(x)
        K = self.embedding_K(x)  # (24, 32, 32, 32)
        V = self.embedding_V(x)

        Q = Q.permute(0, 2, 1, 3).contiguous().reshape(N,T,-1)
        K = K.permute(0, 2, 1, 3).contiguous().reshape(N,T,-1)
        V = V.permute(0, 2, 1, 3).contiguous().reshape(N,T,-1)

        if self.is_pe:
            PE = self.pe(T).repeat(N,1,1).type_as(x)
            Q += PE
            K += PE
            V += PE

        mask = subsequent_mask(T)
        #mask = None

        out, mat = self.attention(Q,K,V,mask)

        if self.inherent:
            self.PA = self.PA.cuda(mat.get_device())
            mat = self.PA * self.alpha  +  mat
            #print(self.alpha)

        #out = out.reshape(N,T,-1,Node)
        #out = out.permute(0,2,1,3).contiguous()

        return out




class Attention(nn.Module):

    def forward(self, query, key, value, m):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        if m is not None:
            m = m.cuda(query.get_device())

            scores = scores.masked_fill(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        p_attn.detach().cpu().numpy()
        return p_val, p_attn


class LocPositionalEncoder(nn.Module):

    def __init__(self, d_model, dout_p, seq_len=300):
        super(LocPositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)

        pos_enc_mat = np.zeros((seq_len, d_model))
        odds = np.arange(0, d_model, 2)
        evens = np.arange(1, d_model, 2)

        for pos in range(seq_len):
            pos_enc_mat[pos, odds] = np.sin(pos / (10000 ** (odds / d_model)))  # 替换pos行，odds列的数据
            pos_enc_mat[pos, evens] = np.cos(pos / (10000 ** (evens / d_model)))

        self.pos_enc_mat = torch.from_numpy(pos_enc_mat).unsqueeze(0)  # (1,3660,d_model)

    def forward(self, S):
        pos = self.pos_enc_mat[:, :S, :]  # 位置矩阵与特征矩阵直接相加
        return pos  # (6,6,C*H*W)

