import torch
import torch.nn as nn
import torch.nn.functional as F
from Aplus.models import *

from einops import rearrange, repeat
from functools import partial
from timm.models.layers import DropPath
# from ssm import SPM,TPM
class SPM(nn.Module):
    def __init__(self, in_joints=6, in_features=32, weight_path = './checkpoint/ck.bin'):
        super().__init__()

        weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
        if weight_path == './checkpoint/ck.bin':
            self.explict_space_prior = weight['model_pos']['spm.explict_space_prior']
            self.explict_space_prior = self.explict_space_prior[[0,1,4,5],:]
            self.explict_space_prior = self.explict_space_prior[:,[0,1,4,5]]
        else:
            self.explict_space_prior = weight
            

        self.explict_space_prior = nn.Parameter(self.explict_space_prior,requires_grad=False)
        
        # self.explict_space_prior = nn.Parameter(torch.eye(4),requires_grad=False)
        
        self.implict_space_prior = nn.Parameter(torch.zeros((in_joints,in_joints)))
        
        self.space_gate = nn.Parameter(torch.zeros((in_joints,in_features)))
        
        self.norm = nn.LayerNorm(in_features)
        
        self.mlp = Mlp(in_features=in_features,hidden_features=in_features*2,out_features=in_features)

    def forward(self, x):
        # x : bs,f,j,d
        
        prior = self.explict_space_prior + self.implict_space_prior
        # prior = self.explict_space_prior
        x_ = x
        x_ = prior@x_
        x_ = self.norm(x_)
        x_ = self.mlp(x_)
        # x = x + x_
        return x_

class TPM(nn.Module):
    def __init__(self, frames=30, in_features=32):
        super().__init__()
        # self.explict_temporal_prior = torch.load('./dataset/temporal_prior_f'+str(frames)+'.pt')
        weight_path = './checkpoint/ck.bin'
        weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
        
        self.explict_temporal_prior = weight['model_pos']['tpm.explict_temporal_prior']
        self.explict_temporal_prior = nn.Parameter(self.explict_temporal_prior,requires_grad=False)

        # self.explict_temporal_prior = torch.load('./dataset/temporal_prior.pt')
        # self.explict_temporal_prior = nn.Parameter(self.explict_temporal_prior, requires_grad=False)
        
        # self.explict_temporal_prior = nn.Parameter(torch.eye(frames), requires_grad=False)
        
        self.implict_temporal_prior = nn.Parameter(torch.zeros((frames,frames)))
        
        self.temporal_gate = nn.Parameter(torch.zeros((frames,in_features)))
        
        self.norm = nn.LayerNorm(in_features)
        
        self.mlp = Mlp(in_features=in_features,hidden_features=in_features*2,out_features=in_features)

    def forward(self, x):
        # x : bs,j,f,d
        # prior = self.explict_temporal_prior + self.implict_temporal_prior
        prior = self.explict_temporal_prior
        x_ = x
        x_ = prior@x_
        x_ = self.norm(x_)
        x_ = self.mlp(x_)
        # x = x + x_
        return x_
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

from typing import List, Tuple

# def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#     t = torch.arange(seq_len, device=freqs.device)
#     freqs = torch.outer(t, freqs).float()
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
#     return freqs_cis

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # freqs: (dim/2,)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # freqs: (seq_len, dim/2)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    # 对应原来的 cos(φ) + i sin(φ)
    cos = freqs.cos()   # (seq_len, dim/2)
    sin = freqs.sin()   # (seq_len, dim/2)

    # 用最后一维存 [real, imag] = [cos, sin]
    freqs_cis = torch.stack((cos, sin), dim=-1)  # (seq_len, dim/2, 2)

    return freqs_cis

# def apply_rotary_emb(
#     xq: torch.Tensor,
#     xk: torch.Tensor,
#     freqs_cis: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
#     xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
#     xq_ = torch.view_as_complex(xq_)
#     xk_ = torch.view_as_complex(xk_)
#     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
#     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
#     return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    xq, xk: (B, T, C)
    freqs_cis: (T, C/2, 2)  # 来自 precompute_freqs_cis
    """

    def _rotate(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape

        # 把最后一维拆成 (C/2, 2)，对应 [real, imag]
        x_ = x.float().reshape(B, T, -1, 2)   # (B, T, C/2, 2)
        x_real = x_[..., 0]                  # (B, T, C/2)
        x_imag = x_[..., 1]                  # (B, T, C/2)

        # freqs_cis: (T, C/2, 2) -> cos, sin
        cos = freqs_cis[..., 0]              # (T, C/2)
        sin = freqs_cis[..., 1]              # (T, C/2)

        # 为了和 (B, T, C/2) 对齐，在 batch 维上 broadcast
        cos = cos.unsqueeze(0)               # (1, T, C/2)
        sin = sin.unsqueeze(0)               # (1, T, C/2)

        # 复数乘法展开：
        # (x_real + i x_imag) * (cos + i sin)
        # = (x_real*cos - x_imag*sin) + i (x_real*sin + x_imag*cos)
        xr = x_real * cos - x_imag * sin
        xi = x_real * sin + x_imag * cos

        x_out = torch.stack((xr, xi), dim=-1)     # (B, T, C/2, 2)
        x_out = x_out.flatten(2)                  # (B, T, C)，和原来形状一样
        return x_out

    xq_out = _rotate(xq, freqs_cis)
    xk_out = _rotate(xk, freqs_cis)

    return xq_out.type_as(xq), xk_out.type_as(xk)



class RO_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # rotation=precompute_freqs_cis(C,N)
        # rotation=rotation.to(x.device)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # q,k=apply_rotary_emb(q,k,rotation)
        b, c, h, w = q.shape
        q = q.permute(0,2,1,3).reshape(b, h, c*w)
        k = k.permute(0,2,1,3).reshape(b, h, c*w)
        rotation=precompute_freqs_cis(c*w,h)
        rotation=rotation.to(x.device)
        q,k=apply_rotary_emb(q,k,rotation)
        w_ = torch.bmm(q, k.permute(0,2,1))
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        v = v.permute(0,2,1,3).reshape(b, h, c*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v.permute(0,2,1), w_).permute(0,2,1)
        h_ = self.proj(h_)
        h_ = self.proj_drop(h_)
        return x+h_

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RO_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
    
            
class Inertial_PoseTransformer(nn.Module):
    def __init__(self, num_frame=9, in_num_joints=17, out_num_joints=24, in_chans=12, out_chans=6, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,norm_layer=None,
                 with_spatial_block = True,
                 with_spatial_pos_embed = True,
                 with_temporal_pos_embed = True,
                 with_ssms = False,
                 with_ssmt = False,
                 ):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.with_spatial_block = with_spatial_block
        self.with_spatial_pos_embed  = with_spatial_pos_embed
        self.with_temporal_pos_embed = with_temporal_pos_embed
        self.with_ssms = with_ssms
        self.with_ssmt = with_ssmt
        if not self.with_spatial_block:
            self.with_spatial_pos_embed = False
            
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * in_num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = out_num_joints * out_chans   #### r6d output 

        if self.with_ssms:
            self.spm = SPM(in_joints=in_num_joints,in_features=embed_dim_ratio)
        
        if self.with_ssmt:
            self.tpm = TPM(frames=num_frame,in_features=embed_dim_ratio)

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)

        if self.with_spatial_pos_embed:
            self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, in_num_joints, embed_dim_ratio))

        if self.with_temporal_pos_embed:
            self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if self.with_spatial_block:
            self.Spatial_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        # self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )


    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)

        if self.with_spatial_pos_embed:
            x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        if self.with_ssms:
            x = rearrange(x,'(b f) j p->b f j p',b=b)
            x = self.spm(x)
            x = rearrange(x,'b f j p->(b f) j p',b=b)

        if self.with_spatial_block:
            for blk in self.Spatial_blocks:
                x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward_features(self, x):
        bs,f,jc = x.shape
        if self.with_temporal_pos_embed:
            x += self.Temporal_pos_embed

        x = self.pos_drop(x)

        if self.with_ssmt:
            x = rearrange(x,'bs f (j c)-> (bs j) f c',j=6)
            x = self.tpm(x)
            x = rearrange(x,'(bs j) f c-> bs f (j c)',j=6)


        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        # x = self.weighted_mean(x)
        # x = x.view(b, 1, -1)
        return x


    def forward(self, x):
        # x: bs, frames, joints, 12

        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x) # x: [256, 180, 544]
        x = self.forward_features(x) # x: [256, 1, 544]

        x = self.head(x) # x: [256, 1, 144]

        # x = x.squeeze(1) 
        # x = x.view(b, 1, p, -1)

        return x

class Inertial_PoseTransformer_Upper(nn.Module):

    def __init__(self, num_frame=30, in_num_joints=4, out_num_joints=10, in_chans=12, out_chans = 6, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,norm_layer=None,
                 with_spatial_block = True,
                 with_spatial_pos_embed = True,
                 with_temporal_pos_embed = True,
                 with_ssms = False,
                 with_ssmt = False,
                 weight_path = './checkpoint/ck.bin'
                 ):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.with_spatial_block = with_spatial_block
        self.with_spatial_pos_embed  = with_spatial_pos_embed
        self.with_temporal_pos_embed = with_temporal_pos_embed
        self.with_ssms = with_ssms
        self.with_ssmt = with_ssmt
        self.in_chans = in_chans
        self.in_num_joints = in_num_joints
        if not self.with_spatial_block:
            self.with_spatial_pos_embed = False
            
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * in_num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = out_num_joints * out_chans  #### r6d output 

        if self.with_ssms:
            self.spm = SPM(in_joints=in_num_joints,in_features=embed_dim_ratio, weight_path=weight_path)
        
        if self.with_ssmt:
            self.tpm = TPM(frames=num_frame,in_features=embed_dim_ratio)

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)

        if self.with_spatial_pos_embed:
            self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, in_num_joints, embed_dim_ratio))

        if self.with_temporal_pos_embed:
            self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if self.with_spatial_block:
            self.Spatial_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        # self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )


    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)

        if self.with_spatial_pos_embed:
            x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        if self.with_ssms:
            x = rearrange(x,'(b f) j p->b f j p',b=b)
            x = self.spm(x)
            x = rearrange(x,'b f j p->(b f) j p',b=b)

        if self.with_spatial_block:
            for blk in self.Spatial_blocks:
                x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward_features(self, x):
        bs,f,jc = x.shape
        if self.with_temporal_pos_embed:
            x += self.Temporal_pos_embed

        x = self.pos_drop(x)

        if self.with_ssmt:
            x = rearrange(x,'bs f (j c)-> (bs j) f c',j=self.in_num_joints)
            x = self.tpm(x)
            x = rearrange(x,'(bs j) f c-> bs f (j c)',j=self.in_num_joints)


        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        # x = self.weighted_mean(x)
        # x = x.view(b, 1, -1)
        return x


    def forward(self, x):
        # x: bs, frames, joints, 12
        x = x.reshape(x.shape[0], x.shape[1], self.in_num_joints, self.in_chans)
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x) # x: [256, 180, 544]
        x = self.forward_features(x) # x: [256, 1, 544]

        x = self.head(x) # x: [256, 1, 144]

        # x = x.squeeze(1) 
        # x = x.view(b, 1, p, -1)

        return x

class FiLM_Positional(nn.Module):
    def __init__(self, in_dim, num_features, hidden_dim=128, num_imus=4, chans=12, dropout=0.2, use_tanh_gamma=True, pe_dim=16):
        super().__init__()
        self.num_features = num_features
        self.num_imus = num_imus
        self.chans = chans
        self.use_tanh_gamma = use_tanh_gamma
        self.shape_pos_embed = nn.Parameter(torch.randn(1, in_dim, pe_dim))

        self.gamma_mlp = nn.Sequential(
            nn.Linear(in_dim * pe_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features)
        )

        self.beta_mlp = nn.Sequential(
            nn.Linear(in_dim * pe_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features)
        )

    def forward(self, x, cond):
        B, T, J, F = x.shape
        cond = cond.unsqueeze(-1)
        cond_pe = cond * self.shape_pos_embed  # [B, T, in_dim, pe_dim]
        cond_pe = cond_pe.flatten(2)  # [B, T, in_dim * pe_dim]

        gamma = self.gamma_mlp(cond_pe)
        beta = self.beta_mlp(cond_pe)

        if self.use_tanh_gamma:
            gamma = torch.tanh(gamma)

        gamma = gamma.view(B, T, J, F)
        beta = beta.view(B, T, J, F)

        return x + gamma * x + beta


class BetaNoiseTransformer(nn.Module):
    def __init__(self, beta_dim, num_features, hidden_dim=128):
        super().__init__()
        self.beta_mlp = nn.Sequential(
            nn.Linear(beta_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features)
        )

    def forward(self, beta):
        return self.beta_mlp(beta)  # [B, T, num_features]

class old_SDN(nn.Module):
    def __init__(
        self, num_frame=30, num_imus=4, chans=12, beta_dim=10, 
        embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2.,
        hidden_dim = 128, use_tanh_gamma=False
    ):
        super(old_SDN, self).__init__()
        self.num_imus = num_imus
        self.chans = chans
        self.film = FiLM_Positional(beta_dim, num_imus * chans,
                         hidden_dim = hidden_dim, use_tanh_gamma = use_tanh_gamma)
        self.denoiser = Inertial_PoseTransformer_Upper(in_num_joints=num_imus, out_num_joints=num_imus, in_chans=chans, out_chans=chans, num_frame=num_frame,
                                                       with_spatial_block = True, with_spatial_pos_embed = True, with_temporal_pos_embed = True,
                                                       with_ssms = True, with_ssmt = True,
                                                       embed_dim_ratio=embed_dim_ratio, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)

    def forward(self, x):
        # x: B, T, num_imus*chans + beta_dim
        imu, beta = x[:, :, :self.num_imus * self.chans], x[:, :, self.num_imus * self.chans:]

        # reshape imu to [B, T, joints, channels]
        imu = imu.view(imu.size(0), imu.size(1), self.num_imus, self.chans)

        # Apply FiLM modulation
        x_mod = self.film(imu, beta)

        # Denoise + shape compensation
        y = self.denoiser(x_mod)
        return y

class SDN(nn.Module):
    def __init__(self, num_frame=30, num_imus=4, chans=12, beta_dim=10,
                 embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2.,
                 hidden_dim=128, use_tanh_gamma=True, use_gate=True,
                 radio=0.5):
        super().__init__()
        self.num_imus = num_imus
        self.chans = chans
        self.num_features = num_imus * chans
        self.use_gate = use_gate
        self.radio = radio

        self.film = FiLM_Positional(beta_dim, self.num_features,
                                    hidden_dim=hidden_dim,
                                    use_tanh_gamma=use_tanh_gamma)

        self.beta_noise = BetaNoiseTransformer(beta_dim, self.num_features, hidden_dim=hidden_dim)

        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(beta_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.lambda_beta = nn.Parameter(torch.tensor(0.3))  # learnable scaling

        self.denoiser = Inertial_PoseTransformer_Upper(
            in_num_joints=num_imus, out_num_joints=num_imus,
            in_chans=chans, out_chans=chans, num_frame=num_frame,
            with_spatial_block=True, with_spatial_pos_embed=True, with_temporal_pos_embed=True,
            with_ssms=True, with_ssmt=True,
            embed_dim_ratio=embed_dim_ratio, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio)

    def forward(self, x):
        imu, beta = x[:, :, :self.num_features], x[:, :, self.num_features:]
        B, T, _ = imu.shape
        imu = imu.view(B, T, self.num_imus, self.chans)

        imu_mod = self.film(imu, beta)
        beta_noise = self.beta_noise(beta).view(B, T, self.num_imus, self.chans)

        y = self.denoiser(imu_mod)+self.radio*beta_noise
        return y

class MP(nn.Module):
    def __init__(
        self, num_frame=30, num_imus=4, chans=12, beta_dim=10, 
        hidden_dim = 128, use_tanh_gamma=True
    ):
        super(MP, self).__init__()
        self.num_imus = num_imus
        self.chans = chans
        self.film = FiLM_Positional(beta_dim, num_imus * chans,
                         hidden_dim = hidden_dim, use_tanh_gamma = use_tanh_gamma)
        self.predictor = Inertial_PoseTransformer_Upper(in_num_joints=num_imus, in_chans=chans, num_frame=num_frame,
                                                       with_spatial_block = True, with_spatial_pos_embed = True, with_temporal_pos_embed = True,
                                                       with_ssms = True, with_ssmt = True)

    def forward(self, x):
        # x: B, T, num_imus*chans + beta_dim
        imu, beta = x[:, :, :self.num_imus * self.chans], x[:, :, self.num_imus * self.chans:]

        # reshape imu to [B, T, joints, channels]
        imu = imu.view(imu.size(0), imu.size(1), self.num_imus, self.chans)

        # Apply FiLM modulation
        x_mod = self.film(imu, beta)

        # predict motion
        y = self.predictor(x_mod)
        return y

# class Gosh(nn.Module):
#     def __init__(self, num_imus=4, chans=12, num_frame=30, out_num_joints = 10, denoise = True, with_ssms=True, with_ssmt=True):
#         super().__init__()
#         self.sdn = SDN(num_imus=num_imus, chans=chans, hidden_dim=64)
#         self.pos = Inertial_PoseTransformer_Upper(num_frame=num_frame,in_num_joints=num_imus, 
#                                                   out_num_joints = out_num_joints, in_chans = chans,
#                                                     with_spatial_block = True,
#                                                     with_spatial_pos_embed = True,
#                                                     with_temporal_pos_embed = True,
#                                                     with_ssms = with_ssms,
#                                                     with_ssmt = with_ssmt)
#         self.denoise = denoise
#         self.num_imus = num_imus
#         self.chans = chans
#     def forward(self, x):
#         if self.denoise:
#             denoised_x = x[:,:,:48]-self.sdn(x)
#         else:
#             denoised_x = x[:, :, :self.num_imus * self.chans]
#         pr_pose = self.pos(denoised_x)
#         return pr_pose


class Gosh(nn.Module):
    def __init__(self, num_imus=4, chans=12, num_frame=30, out_num_joints = 10, denoise = True, with_ssms=True, with_ssmt=True):
        super().__init__()
        self.sdn = SDN(num_imus=num_imus, chans=chans, hidden_dim=64)
        self.pos = Inertial_PoseTransformer_Upper(num_frame=num_frame,in_num_joints=num_imus, 
                                                  out_num_joints = out_num_joints, in_chans = chans,
                                                    with_spatial_block = True,
                                                    with_spatial_pos_embed = True,
                                                    with_temporal_pos_embed = True,
                                                    with_ssms = with_ssms,
                                                    with_ssmt = with_ssmt)
        self.denoise = denoise
        self.num_imus = num_imus
        self.chans = chans
    def forward(self, x):
        if self.denoise:
            denoised_x = x[:,:,:48]-self.sdn(x)
        else:
            denoised_x = x[:, :, :self.num_imus * self.chans]
        pr_pose = self.pos(denoised_x)
        return pr_pose


# class PMP(nn.Module):
#     def __init__(self, num_imus=4, chans=12, num_frame=30, out_num_joints = 10, with_ssms=True, with_ssmt=True):
#         super().__init__()
#         self.film = FiLM(10, num_imus * chans)
#         self.pose = Inertial_PoseTransformer_Upper(num_frame=num_frame,in_num_joints=num_imus, 
#                                                    out_num_joints = out_num_joints, in_chans = chans,
#                                                    with_spatial_block = True,
#                                                    with_spatial_pos_embed = True,
#                                                    with_temporal_pos_embed = True,
#                                                    with_ssms = with_ssms,
#                                                    with_ssmt = with_ssmt)
#         self.num_imus = num_imus
#         self.chans = chans
#     def forward(self, x):
#         # print(x.shape)
#         imu, beta = x[:, :, :self.num_imus * self.chans], x[:, :, self.num_imus * self.chans:]
#         imu = imu.view(imu.size(0), imu.size(1), self.num_imus, self.chans)
#         x = self.film(imu, beta)
#         pr_pose = self.pose(x)
#         return pr_pose




