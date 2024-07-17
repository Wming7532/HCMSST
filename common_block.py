import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-7

def linear_attn(q, k, x, EPS=EPS):

    k = k.transpose(-2, -1)
    x = q @ (k @ x)
    q = q @ k.sum(dim=-1, keepdim=True) + EPS
    x = x / q
    return x


class ACT():
    r'''
    args:
        actType: chose one of 'relu', 'prelu', 'lrelu'
        negative_slope: for 'lrelu' and initial vlaue for 'prelu'
    return:
        activation function
    '''

    def __init__(self, actType, negative_slope=0.01):
        super().__init__()
        self.actType = actType
        self.negative_slope = negative_slope

    def get_act(self, ):
        if self.actType.lower() == 'relu':
            act = nn.ReLU(True)
        elif self.actType.lower() == 'lrelu':
            act = nn.LeakyReLU(self.negative_slope)
        elif self.actType.lower() == 'prelu':
            act = nn.PReLU()
        elif self.actType.lower() == 'gelu':
            act = nn.GELU()
        else:
            raise ('This type of %s activation is not added in ACT, please add it first.' % self.actType)
        return act

class speMultiAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.num_heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.scale = dim ** -0.5
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_in, padsize=0):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """

        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        if x.shape[1] % self.num_heads != 0:
            padsize = (x.shape[1] // self.num_heads + 1) * self.num_heads - x.shape[1]
            x = F.pad(x, [0, 0, 0, padsize], mode='replicate')
        qkv = self.qkv(x).reshape(b, -1, self.num_heads, 3, c).permute(3, 0, 2, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(-2, -1).reshape(b, -1, c)
        if padsize:
            x = x[:, :-padsize, :]
        out_c = self.proj(x).reshape(b, h, w, c)
        out = out_c

        return out

class spaMultiAttn(nn.Module):
    def __init__(
            self,
            embed_dim,
            heads,
            patchSize
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = embed_dim // heads
        self.patchSize = patchSize
        self.embed_dim = embed_dim
        self.relu = nn.ReLU()
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.scale = head_dim ** -0.5
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x_in, padsize=0):
        """
        x_in: [b,h,w,d]
        return out: [b,h,w,d]
        """
        b, h, w, d = x_in.shape
        x = x_in.reshape(b, h * w, d)
        if x.shape[2] % self.num_heads != 0:
            padsize = (x.shape[2] // self.num_heads + 1) * self.num_heads - x.shape[2]
            x = F.pad(x, [0, padsize, 0, 0], mode='replicate')
        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, d // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = q.softmax(dim=-2), k.softmax(dim=-2)
        x = linear_attn(q, k, v)
        x = x.transpose(1, 2).reshape(b, h * w, d)
        if padsize != 0:
            x = x[:, :, :-padsize]
        out_s = self.proj(x).reshape(b, h, w, d)
        out = out_s

        return out
