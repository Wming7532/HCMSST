import torch.nn as nn
import torch
import torch.nn.functional as F
from fractions import Fraction
from .common_block import ACT, spaMultiAttn, speMultiAttn


EPS = 1e-7
class FeedForward(nn.Module):
    def __init__(self, dim):
        super(FeedForward, self).__init__()
        act = ACT('gelu')
        self.conv1 = nn.Conv2d(dim, dim*4, 1)
        self.gelu1 = act.get_act()
        self.conv2 = nn.Conv2d(dim*4, dim, 1)

    def forward(self, x):
        x = self.conv1(x.permute(0, 3, 1, 2))
        x = self.gelu1(x)
        x = self.conv2(x)
        return x.permute(0, 2, 3, 1)


def down_conv(in_features, out_features, idx, U_number):

    conv_layers = nn.ModuleList([
        nn.Conv2d(
            in_features,
            out_features,
            kernel_size=2 ** i if i <= U_number // 2 else 2 ** (U_number - 1 - i),
            stride=2 ** i if i <= U_number // 2 else 2 ** (U_number - 1 - i)
        )
        for i in range(U_number)
    ])

    return conv_layers[idx]


def up_conv(in_features, out_features, idx, U_number):

    T_conv_layers = nn.ModuleList([
        nn.ConvTranspose2d(
            in_features,
            out_features,
            kernel_size=2 ** i if i <= U_number // 2 else 2 ** (U_number - 1 - i),
            stride=2 ** i if i <= U_number // 2 else 2 ** (U_number - 1 - i)
        )
        for i in range(U_number)
    ])

    return T_conv_layers[idx]


class SpaTransBlock(nn.Module):
    def __init__(self, embed_feats, hidden_features, numHeads, patchSize, inner_numLayers, spa_idx, U_number=5,
                 ) -> None:
        super().__init__()
        self.inner_numLayers = inner_numLayers
        self.embed_feats = embed_feats
        self.hidden_features = hidden_features
        self.spa_idx = spa_idx

        self.embed_conv = down_conv(embed_feats, hidden_features, spa_idx, U_number)
        self.unEmbed_conv = up_conv(hidden_features, embed_feats, spa_idx, U_number)
        self.layerNorm = nn.LayerNorm(hidden_features)
        self.block = nn.ModuleList([])
        for _ in range(self.inner_numLayers):
            self.block.append(nn.ModuleList([
                nn.LayerNorm(self.hidden_features),
                spaMultiAttn(self.hidden_features, numHeads, patchSize),
                FeedForward(self.hidden_features)
            ]))

    def forward(self, x_in):
        """
        input: x  shape [B, C, H, W]
        """
        x = self.embed_conv(x_in)
        x = self.layerNorm(x.permute(0, 2, 3, 1))
        for norm, multiAttn, ffn in self.block:
            x = x + multiAttn(norm(x))
            x = x + ffn(norm(x))
        x = self.layerNorm(x)
        x = self.unEmbed_conv(x.permute(0, 3, 1, 2))
        return x


class SpecTransBlock(nn.Module):
    def __init__(
            self,
            embed_feats, hidden_features, numHeads, inner_numLayers) -> None:
        super().__init__()
        self.inner_numLayers = inner_numLayers
        self.hidden_features = hidden_features
        self.embed_feats = embed_feats

        self.embed_conv = nn.Conv2d(embed_feats, hidden_features, 1)
        self.unEmbed_conv = nn.Conv2d(hidden_features, embed_feats, 3, 1, 1)
        self.layerNorm = nn.LayerNorm(hidden_features)
        self.block = nn.ModuleList([])
        for _ in range(self.inner_numLayers):
            self.block.append(nn.ModuleList([
                nn.LayerNorm(hidden_features),
                speMultiAttn(self.hidden_features, numHeads),
                FeedForward(self.hidden_features)
            ]))

    def forward(self, x_in):
        """
        input: x  shape [B, C, H, W]
        """
        x = self.embed_conv(x_in)
        x = self.layerNorm(x.permute(0, 2, 3, 1))
        for norm, multiAttn, ffn in self.block:
            x = x + multiAttn(norm(x))
            x = x + ffn(norm(x))
        x = self.layerNorm(x)
        x = self.unEmbed_conv(x.permute(0, 3, 1, 2))
        return x
        # pass

#
class SpatialAttention(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels//4, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels//4, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha*feat_e + x

        return out

class ChannelAttention(nn.Module):
    """Channel attention module"""

    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta*feat_e + x

        return out

class SCDRB(nn.Module):
    def __init__(self, opt, recursive_block_num):
        super(SCDRB, self).__init__()
        self.convDim = opt['convDim']
        self.scale = opt['scale']
        self.feature_num = self.convDim//4
        self.recursive_block_num = recursive_block_num
        act = ACT('relu')
        self.residual_block = nn.Sequential(
            nn.Conv2d(self.convDim, self.feature_num, 1),
            act.get_act(),
            nn.Conv2d(self.feature_num, self.feature_num, 3, 1, 1),
            act.get_act(),
            nn.Conv2d(self.feature_num, self.convDim, 1)
        )
        self.RB1 = nn.ModuleList()
        self.RB2 = nn.ModuleList()
        for _ in range(self.recursive_block_num):
            self.RB1.append(self.residual_block)
            self.RB2.append(self.residual_block)
        self.pam_ms = nn.ModuleList(SpatialAttention(self.convDim) for _ in range(self.recursive_block_num))  # 空间注意力机制
        self.pam_hs = nn.ModuleList(ChannelAttention() for _ in range(self.recursive_block_num))  # 通道注意力机制
        self.skip_ms = nn.ModuleList([nn.Sequential(nn.Conv2d(self.convDim*5, self.convDim, 1),
                                                    nn.PReLU()) if _ >= 2
                                      else nn.Sequential(nn.Conv2d(self.convDim*3, self.convDim, 1),
                                                         nn.PReLU())
                                      for _ in range(self.recursive_block_num)])
        self.skip_hs = nn.ModuleList([nn.Sequential(nn.Conv2d(self.convDim*5, self.convDim, 1),
                                                    nn.PReLU()) if _ >= 2
                                      else nn.Sequential(nn.Conv2d(self.convDim*3, self.convDim, 1),
                                                         nn.PReLU())
                                      for _ in range(self.recursive_block_num)])
        self.ms_recons_layer = nn.Sequential(
            nn.Conv2d(self.convDim*self.recursive_block_num, self.convDim, 1),
            nn.ReLU())
        self.hs_recons_layer = nn.Sequential(
            nn.Conv2d(self.convDim*self.recursive_block_num, self.convDim, 1),
            nn.ReLU())

    def forward(self, ms, hs):
        ms_out_block = ms
        hs_out_block = hs
        intermp = []
        interhp = []
        ms_inner_block = []
        hs_inner_block = []
        ms_dense_block = []
        hs_dense_block = []
        for idx in range(self.recursive_block_num):
            ms_inner_out = self.RB1[idx](ms_out_block)
            hs_inner_out = self.RB2[idx](hs_out_block)
            ms_inner_block.append(ms_inner_out)
            hs_inner_block.append(hs_inner_out)
            tmp = F.interpolate(ms_inner_block[idx-1] if idx > 0 else ms, scale_factor=1.0/self.scale, mode='bicubic')
            interpHS = F.interpolate(hs_inner_block[idx-1] if idx > 0 else hs, scale_factor=self.scale, mode='bicubic')
            mp = self.pam_ms[idx](tmp)
            hp = self.pam_hs[idx](interpHS)
            intermp.append(mp)
            interhp.append(hp)
            if idx >= 2:
                ms_out_block = self.skip_ms[idx](torch.cat((ms_inner_out, ms_inner_block[idx-2], ms,
                                                            hp, interhp[idx-2]), dim=1))
                hs_out_block = +self.skip_hs[idx](torch.cat((hs_inner_out, hs_inner_block[idx-2], hs,
                                                            mp, intermp[idx-2]), dim=1))
            else:
                ms_out_block = self.skip_ms[idx](torch.cat((ms_inner_out, ms, hp), dim=1))
                hs_out_block = self.skip_hs[idx](torch.cat((hs_inner_out, hs, mp), dim=1))
            ms_dense_block.append(ms_out_block)
            hs_dense_block.append(hs_out_block)

        ms = self.ms_recons_layer(torch.cat(ms_dense_block, dim=1))+ms
        hs = self.hs_recons_layer(torch.cat(hs_dense_block, dim=1))+hs
        return ms, hs


class Net(nn.Module):
    r"""
    args:
        convDim: dim for spectral embedding
        scale: used to multiply convDim to get the dim for spatial embedding
        numHeads: the head number for multihead attention
        patchSize: the size of patch attention base transformer in spatial transformer
    """

    def __init__(self, opt):
        super().__init__()
        ################  {network args}  ################
        self.useCheckpoint = True
        self.dimHs = opt['LRdim']  # dimHs
        self.dimMs = opt['REFdim']  # dimMs
        self.scale = opt['scale']
        self.numLayers = opt['numLayers']
        self.numHeads = opt['numHeads']
        self.convDim = opt['convDim']
        self.patchSize = opt['patchSize']
        self.embedDim = self.convDim//2
        self.Unet_numLayers = 5
        self.opt = opt
        self.recursive_block_num = 5

        ################  {net module}  ################
        self.conv_head = nn.Conv2d(self.dimHs+self.dimMs, self.convDim, 3, 1, 1)
        self.spe_embed = nn.Conv2d(self.dimHs+self.dimMs, self.convDim, 3, 1, 1)
        self.SCDRB = SCDRB(self.opt, self.recursive_block_num)

        def calculate_spe_dim(idx):
            if idx <= self.Unet_numLayers // 2:
                return int(self.convDim * Fraction(4 - idx, 3))
            else:
                return int(self.convDim * Fraction(idx, 3))

        self.spa_trans = nn.ModuleList([
            SpaTransBlock(
                self.convDim,
                self.convDim,
                self.numHeads,
                self.patchSize,
                inner_numLayers=self.numLayers,
                spa_idx=_,
                U_number=self.Unet_numLayers
            ) for _ in range(self.Unet_numLayers)
        ])
        self.spe_trans = nn.ModuleList([
            SpecTransBlock(
                self.convDim,
                calculate_spe_dim(_),
                self.numHeads,
                inner_numLayers=self.numLayers,
            ) for _ in range(self.Unet_numLayers)
        ])
        self.ms_skip = nn.ModuleList([nn.Conv2d(self.convDim * 2, self.convDim, 1) for _ in range(self.Unet_numLayers//2)])
        self.hs_skip = nn.ModuleList([nn.Conv2d(self.convDim * 2, self.convDim, 1) for _ in range(self.Unet_numLayers//2)])
        self.fusion_ms = nn.ModuleList([nn.Conv2d(self.convDim * 2, self.convDim, 1)
                                        for _ in range(self.Unet_numLayers//2+1)])
        self.fusion_hs = nn.ModuleList([nn.Conv2d(self.convDim * 2, self.convDim, 1)
                                        for _ in range(self.Unet_numLayers//2+1)])
        self.conv_ms = nn.Sequential(nn.Conv2d(self.convDim * (self.Unet_numLayers//2+1), self.convDim, 1),
                                     nn.GELU())
        self.conv_hs = nn.Sequential(nn.Conv2d(self.convDim * (self.Unet_numLayers//2+1), self.convDim, 1),
                                     nn.GELU())
        self.upsample = nn.Sequential(
            nn.Conv2d(self.convDim, self.embedDim, 1),
            nn.GELU(),
            nn.Conv2d(self.embedDim, self.scale**2*self.embedDim, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=self.scale),
            nn.Conv2d(self.embedDim, self.convDim, 3, 1, 1)
        )
        self.dense_fuse = nn.Sequential(
            nn.Conv2d(self.convDim * 2, self.convDim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(self.convDim, self.dimHs, 3, 1, 1)
        )

    def forward(self, batchData, mask=None):
        """
        inputs:
            hs: image with the size of [B, C1, h, w]
            hs: image with the size of [B, C2, H, W]
        """

        hs = batchData['LR']
        ms = batchData['REF']
        del batchData
        interpHS = F.interpolate(hs, scale_factor=self.scale, mode='bicubic', align_corners=False)
        tmp = F.interpolate(ms, scale_factor=1.0 / self.scale, mode='bicubic', align_corners=False)
        ms = self.conv_head(torch.cat((ms, interpHS), dim=1))
        # B, C, H, W = ms.shape
        hs = self.spe_embed(torch.cat((tmp, hs), dim=1))
        # B, c, h, w = hs.shape
        ms, hs = self.SCDRB(ms, hs)
        ms_inner_block = []
        hs_inner_block = []

        fuse_fa = []
        fuse_f = []
        for idx in range(self.Unet_numLayers):
            ms = self.spa_trans[idx](ms)
            hs = self.spe_trans[idx](hs)

            if idx <= self.Unet_numLayers // 2:
                ms_inner_block.append(ms)
                hs_inner_block.append(hs)
            else:
                tmpMs = ms_inner_block[idx - 2 * (idx // 2)]
                tmpHs = hs_inner_block[idx - 2 * (idx // 2)]
                ms_inner_block.append(self.ms_skip[idx - (self.Unet_numLayers // 2 + 1)](torch.cat((tmpMs, ms), dim=1)))
                hs_inner_block.append(self.hs_skip[idx - (self.Unet_numLayers // 2 + 1)](torch.cat((tmpHs, hs), dim=1)))

            if idx >= self.Unet_numLayers // 2 and idx != 0:
                tmp_Ms = F.interpolate(ms_inner_block[idx - self.Unet_numLayers // 2], scale_factor=1 / self.scale,
                                       mode='bicubic')
                tmp_Hs = F.interpolate(hs_inner_block[idx - self.Unet_numLayers // 2], scale_factor=self.scale,
                                       mode='bicubic')
                tmpMs = self.fusion_ms[idx - self.Unet_numLayers // 2](torch.cat((tmp_Hs, ms_inner_block[idx]), dim=1))
                tmpHs = self.fusion_hs[idx - self.Unet_numLayers // 2](torch.cat((tmp_Ms, hs_inner_block[idx]), dim=1))
                ms = tmpMs + ms_inner_block[idx]
                hs = tmpHs + hs_inner_block[idx]

                if idx < self.Unet_numLayers - 1:
                    fuse_fa.append(tmpMs)
                    fuse_f.append(tmpHs)
                else:
                    fuse_fa.append(ms)
                    fuse_f.append(hs)
            else:
                ms = ms_inner_block[idx]
                hs = hs_inner_block[idx]

            if self.Unet_numLayers // 2 == 0:
                fuse_fa.append(ms)
                fuse_f.append(hs)

        tmpMs = self.conv_ms(torch.cat(fuse_fa, dim=1))
        tmpHs = self.conv_hs(torch.cat(fuse_f, dim=1))
        tmp_Hs = self.upsample(tmpHs)
        tmp = self.dense_fuse(torch.cat((tmpMs, tmp_Hs), dim=1))
        ms = tmp + interpHS

        return ms

    def loss(self, rec=None, gt=None):
        return F.l1_loss(rec, gt['GT'])