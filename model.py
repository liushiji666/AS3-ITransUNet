## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from nihao import MixBlock

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample1(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample1, self).__init__(*m)

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,  in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(self.in_chans, self.embed_dim, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x
##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=7,
                 out_channels=5,
                 dim=48,
                 num_blocks=[1, 1, 1, 1],
                 num_refinement_blocks=4,
                 heads=[2, 2, 2, 2],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer, self).__init__()
        
        upscale = 2
        self.nearest = nn.Upsample(scale_factor=upscale, mode='nearest')
        self.patch_embed_enc1 = PatchEmbed(dim, dim * 2 ** 1)
        self.patch_embed_enc2 = PatchEmbed(dim, dim)
        self.patch_embed_enc3 = PatchEmbed(dim * 2 ** 1, dim * 2 ** 1)
        self.patch_embed_la = PatchEmbed(dim, dim)
        self.patch_embed_dec3 = PatchEmbed(dim * 2 ** 1, dim * 2 ** 1)
        self.patch_embed_dec2 = PatchEmbed(dim, dim)
        self.patch_embed_dec1 = PatchEmbed(dim* 2 ** 1, dim* 2 ** 1)
        self.patch_embed_ref = PatchEmbed(dim, dim)
        self.patch_unembed1 = PatchUnEmbed(dim * 2 ** 1)
        self.patch_unembed2 = PatchUnEmbed(dim)
        self.patch_unembed3 = PatchUnEmbed(dim//2)

        self.patch_embed1_3 = OverlapPatchEmbed(inp_channels, out_channels)
        self.patch_embed1_2 = OverlapPatchEmbed(inp_channels, out_channels - 2)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.gamma = nn.Parameter(torch.ones(6))

        self.encoder_level1 = nn.ModuleList([
            MixBlock(dim=dim * 2 ** 1, num_heads=heads[0], window_size=16,
                     dwconv_kernel_size=3, shift_size=0, mlp_ratio=4.,
                     qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=drop_path[i] if isinstance(0., list) else 0.,
                     norm_layer=nn.LayerNorm)
            for i in range(num_blocks[0])])

        # self.encoder_level1 = nn.Sequential(*[
        #     TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
        #                      LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim * 2 ** 1)  ## From Level 1 to Level 2
        self.down2_3 = Downsample(dim * 2 ** 1)
        self.down3_4 = Downsample(dim * 2 ** 1)
        self.down_re = Downsample(dim * 2 ** 1)
        self.encoder_level2 = nn.ModuleList([
            MixBlock(dim=int(dim), num_heads=heads[1], window_size=16,
                     dwconv_kernel_size=3, shift_size=0, mlp_ratio=4.,
                     qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=drop_path[i] if isinstance(0., list) else 0.,
                     norm_layer=nn.LayerNorm) for i in range(num_blocks[1])])

        self.encoder_level3 = nn.ModuleList([
            MixBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], window_size=16,
                     dwconv_kernel_size=3, shift_size=0, mlp_ratio=4.,
                     qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=drop_path[i] if isinstance(0., list) else 0.,
                     norm_layer=nn.LayerNorm) for i in range(num_blocks[1])])
        # self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.latent = nn.ModuleList([
            MixBlock(dim=int(dim), num_heads=heads[2], window_size=16,
                     dwconv_kernel_size=3, shift_size=0, mlp_ratio=4.,
                     qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=drop_path[i] if isinstance(0., list) else 0.,
                     norm_layer=nn.LayerNorm) for i in range(num_blocks[2])])

        self.reduce_chan_level1i = nn.Conv2d(int(dim * 2), int(dim * 2 ** 1), kernel_size=3, stride=1, padding=1, bias=bias)
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2 ** 2), int(dim* 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 1), int(dim), kernel_size=1, bias=bias)
        self.reduce_chan_level2i = nn.Conv2d(int(dim), int(dim), kernel_size=3, stride=1, padding=1, bias=bias)
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level3i = nn.Conv2d(int(dim * 2), int(dim * 2 ** 1), kernel_size=3, stride=1, padding=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            MixBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], window_size=16,
                     dwconv_kernel_size=3, shift_size=0, mlp_ratio=4.,
                     qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=drop_path[i] if isinstance(0., list) else 0.,
                     norm_layer=nn.LayerNorm) for i in range(num_blocks[1])])

        self.decoder_level2 = nn.ModuleList([
            MixBlock(dim=int(dim), num_heads=heads[1], window_size=16,
                     dwconv_kernel_size=3, shift_size=0, mlp_ratio=4.,
                     qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=drop_path[i] if isinstance(0., list) else 0.,
                     norm_layer=nn.LayerNorm) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.up2_3 = Upsample(int(dim))

        self.up4_3 = Upsample(int(dim))

        self.decoder_level1 = nn.ModuleList([
            MixBlock(dim=int(dim* 2 ** 1), num_heads=heads[0], window_size=16,
                     dwconv_kernel_size=3, shift_size=0, mlp_ratio=4.,
                     qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=drop_path[i] if isinstance(0., list) else 0.,
                     norm_layer=nn.LayerNorm) for i in range(num_blocks[0])])

        self.refinement = nn.ModuleList([
            MixBlock(dim=int(dim), num_heads=heads[0], window_size=16,
                     dwconv_kernel_size=3, shift_size=0, mlp_ratio=4.,
                     qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=drop_path[i] if isinstance(0., list) else 0.,
                     norm_layer=nn.LayerNorm) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output1 = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output2 = nn.Conv2d(int(dim), out_channels - 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output1i = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output2i = nn.Conv2d(int(dim), out_channels - 2, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.upsample1 = Upsample1(upscale, out_channels)

    def forward(self, inp_img, i, localFeats_dec3, localFeats_dec2, localFeats_dec1):
        inp_img1 = self.nearest(inp_img)
        if (i == 0)|(i == 6):
            inp_img1 = self.patch_embed1_2(inp_img1)
        else:
            inp_img1 = self.patch_embed1_3(inp_img1)
        inp_enc_level1 = self.patch_embed(inp_img)
        b, c, H, W = inp_enc_level1.shape
        inp_enc_level1 = self.patch_embed_enc1(inp_enc_level1)
        for blk in self.encoder_level1:
            blk.H, blk.W = H, W
            inp_enc_level1 = blk(inp_enc_level1, None)
        out_enc_level1 = self.patch_unembed1(inp_enc_level1, (H, W))
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        b, c, H, W = inp_enc_level2.shape
        # print('111', inp_enc_level2.shape)
        inp_enc_level2 = self.patch_embed_enc2(inp_enc_level2)
        # print('222', inp_enc_level2.shape)
        for blk in self.encoder_level2:
            blk.H, blk.W = H, W
            inp_enc_level2 = blk(inp_enc_level2, None)
        out_enc_level2 = self.patch_unembed2(inp_enc_level2, (H, W))
        # out_enc_level2 = self.patch_unembed1(inp_enc_level1, (H, W))

        inp_enc_level3 = self.up2_3(out_enc_level2)
        b, c, H, W = inp_enc_level3.shape
        # print('111', inp_enc_level2.shape)
        inp_enc_level3 = self.patch_embed_enc3(inp_enc_level3)
        # print('222', inp_enc_level2.shape)
        for blk in self.encoder_level3:
            blk.H, blk.W = H, W
            inp_enc_level3 = blk(inp_enc_level3, None)
        out_enc_level3 = self.patch_unembed1(inp_enc_level3, (H, W))

        inp_enc_level4 = self.down3_4(out_enc_level3)
        b, c, H, W = inp_enc_level4.shape
        latent = self.patch_embed_la(inp_enc_level4)
        for blk in self.latent:
            blk.H, blk.W = H, W
            latent = blk(latent, None)
        latent = self.patch_unembed2(latent, (H, W))

        inp_dec_level3 = self.up4_3(latent)
        if i == 0:
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
            inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
            localFeats_dec3 = inp_dec_level3
        else:
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
            inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
            err = inp_dec_level3 - localFeats_dec3
            err = self.reduce_chan_level3i(err)
            inp_dec_level3 = inp_dec_level3 + err
            localFeats_dec3 = inp_dec_level3
        b, c, H, W = inp_dec_level3.shape
        inp_dec_level3 = self.patch_embed_dec3(inp_dec_level3)
        for blk in self.decoder_level3:
            blk.H, blk.W = H, W
            inp_dec_level3 = blk(inp_dec_level3, None)
        out_dec_level3 = self.patch_unembed1(inp_dec_level3, (H, W))

        # inp_dec_level2 = out_dec_level3
        out_dec_level3 = self.down2_3(out_dec_level3)
        if i == 0:
            outlocal_dec2 = self.output2i(out_dec_level3)
            inp_dec_level2 = torch.cat([out_dec_level3, out_enc_level2], 1)
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
            localFeats_dec2 = inp_dec_level2
        else:
            inp_dec_level2 = torch.cat([out_dec_level3, out_enc_level2], 1)
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
            err = inp_dec_level2 - localFeats_dec2
            err = self.reduce_chan_level2i(err)
            inp_dec_level2 = inp_dec_level2 + err
            if (i == 6):
                outlocal_dec2 = self.output2i(inp_dec_level2)
            else:
                outlocal_dec2 = self.output1i(inp_dec_level2)
            localFeats_dec2 = inp_dec_level2
        b, c, H, W = inp_dec_level2.shape
        inp_dec_level2 = self.patch_embed_dec2(inp_dec_level2)
        for blk in self.decoder_level2:
            blk.H, blk.W = H, W
            inp_dec_level2 = blk(inp_dec_level2, None)
        out_dec_level2 = self.patch_unembed2(inp_dec_level2, (H, W))
        # out_dec_level2 = self.decoder_level2(inp_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        if i == 0:
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
            inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
            localFeats_dec1 = inp_dec_level1
        else:
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
            inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
            err = inp_dec_level1 - localFeats_dec1
            err = self.reduce_chan_level1i(err)
            inp_dec_level1 = inp_dec_level1 + err
            localFeats_dec1 = inp_dec_level1
        b, c, H, W = inp_dec_level1.shape
        inp_dec_level1 = self.patch_embed_dec1(inp_dec_level1)
        for blk in self.decoder_level1:
            blk.H, blk.W = H, W
            inp_dec_level1 = blk(inp_dec_level1, None)
        out_dec_level1 = self.patch_unembed1(inp_dec_level1, (H, W))
        # out_dec_level1 = self.decoder_level1(inp_dec_level1)
        inp_dec_level1 = self.down_re(out_dec_level1)
        b, c, H, W = inp_dec_level1.shape
        out_dec_level1 = self.patch_embed_ref(inp_dec_level1)
        for blk in self.refinement:
            blk.H, blk.W = H, W
            out_dec_level1 = blk(out_dec_level1, None)
        out_dec_level1 = self.patch_unembed2(out_dec_level1, (H, W))
        # out_dec_level1 = self.refinement(out_dec_level1)
        if (i == 0)|(i == 6):
            # print('jjj',(self.output2(out_dec_level1)).shape)
            out_dec_level1 = self.output2(out_dec_level1) + inp_img1
        else:
            out_dec_level1 = self.output1(out_dec_level1) + inp_img1
        # out_dec_level1 = self.output(out_dec_level1) + inp_img1
        # out_dec_level1 = self.upsample1(out_dec_level1)
        return out_dec_level1,outlocal_dec2 ,localFeats_dec3, localFeats_dec2, localFeats_dec1