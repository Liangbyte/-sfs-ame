import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import os, sys

try:
    import pywt
except Exception:
    pywt = None


class HaarDWT(nn.Module):
    """Single-level 2D Discrete Wavelet Transform (Haar) implemented in PyTorch.
    Input: (N, 1, H, W) or (N, C, H, W) where H, W divisible by 2.
    Output: (LL, LH, HL, HH) each with shape (N, 1, H/2, W/2) if input is single channel.
    If multi-channel, returns per-channel lists concatenated along channel dim.
    """
    def __init__(self):
        super().__init__()
        # Haar filters
        ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        lh = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
        hl = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
        hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        self.register_buffer('k_ll', ll.view(1, 1, 2, 2))
        self.register_buffer('k_lh', lh.view(1, 1, 2, 2))
        self.register_buffer('k_hl', hl.view(1, 1, 2, 2))
        self.register_buffer('k_hh', hh.view(1, 1, 2, 2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n, c, h, w = x.shape
        # apply per-channel depthwise conv with stride=2 to produce 128x128 bands from 256x256 input
        k_ll = self.k_ll.repeat(c, 1, 1, 1)
        k_lh = self.k_lh.repeat(c, 1, 1, 1)
        k_hl = self.k_hl.repeat(c, 1, 1, 1)
        k_hh = self.k_hh.repeat(c, 1, 1, 1)
        ll = F.conv2d(x, k_ll, stride=2, padding=0, groups=c)
        lh = F.conv2d(x, k_lh, stride=2, padding=0, groups=c)
        hl = F.conv2d(x, k_hl, stride=2, padding=0, groups=c)
        hh = F.conv2d(x, k_hh, stride=2, padding=0, groups=c)
        return ll, lh, hl, hh


class DualStreamVITLSNet(nn.Module):
    """Wrapper that fuses frozen ViT backbone features with LSNet-t features from grayscale DWT high-frequency bands.

    - Keeps provided ViT model frozen (as constructed outside).
    - Builds an LSNet-t branch processing grayscale image via single-level Haar DWT (use LH/HL/HH bands).
    - Concatenates ViT's CLS features with LSNet pooled features, then passes through a fusion MLP and final classifier.
    - Exposes a forward compatible with existing training code: returns (logits, moe_loss) optionally passthrough xray when asked.
    """
    def __init__(self, vit_model: nn.Module, num_classes: int = 2, img_size: int = 256,
                 gn_groups: int = 32, cross_alpha_v: float = 0.2, cross_alpha_f: float = 0.2,
                 use_rgb_dwt: bool = False):
        super().__init__()
        self.vit = vit_model
        self.cross_alpha_v = float(cross_alpha_v)
        self.cross_alpha_f = float(cross_alpha_f)
        self.rgb_dwt = bool(use_rgb_dwt)
        # vit is expected to already be mostly frozen via its freeze_stages; ensure safety
        for n, p in self.vit.named_parameters():
            if p.requires_grad is False:
                continue
            # Keep LoRA, head, fusion, xray trainable per vit.freezing policy
            pass

        # Frequency CNN branch broken into explicit layers to tap the 3rd conv output
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.GroupNorm(gn_groups, 64), nn.ReLU(inplace=True)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(gn_groups, 128), nn.ReLU(inplace=True)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1), nn.GroupNorm(gn_groups, 256), nn.ReLU(inplace=True)
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 2, 1), nn.GroupNorm(gn_groups, 384), nn.ReLU(inplace=True)
        )
        print('DualStream: Using explicit freq CNN with taps after conv3 for cross-attn.')

        self.dwt = HaarDWT()
        if self.rgb_dwt:
            self.rgb_reduce = nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # Determine feature dims
        vit_feat_dim = getattr(self.vit, 'embed_dim', None) or getattr(self.vit, 'num_features', 768)
        lsnet_feat_dim = 384

        # Project LSNet output to a fixed dim if needed
        self.ls_project = nn.Sequential(nn.Flatten(), nn.Linear(384, 384))

        # Bi-directional cross-attention modules (pre-norm) between ViT tokens (after block7) and freq tokens (after conv3)
        # 使用与 ViT 一致的 pre-norm：LN→Attn→残差，LN→MLP→残差
        from timm.models.layers import DropPath, Mlp
        self.mlp_ratio = 4.0
        self.drop = 0.0
        self.drop_path_rate = 0.0
        self.norm_v_attn = nn.LayerNorm(vit_feat_dim)
        self.norm_v_mlp = nn.LayerNorm(vit_feat_dim)
        self.norm_f_attn = nn.LayerNorm(vit_feat_dim)  # 频域先投影到 ViT 维度后再 LN
        self.norm_f_mlp = nn.LayerNorm(vit_feat_dim)
        self.proj_freq2vit = nn.Linear(256, vit_feat_dim)
        # 用于把更新后的频域 tokens 从 ViT 维度投回卷积分支通道数（256）以继续 c4
        self.proj_vit2freq_tokens = nn.Linear(vit_feat_dim, 256)
        # 兼容旧版 PyTorch：某些版本不支持 batch_first 参数
        num_heads = max(1, vit_feat_dim // 64)
        try:
            self.mha = nn.MultiheadAttention(vit_feat_dim, num_heads=num_heads, batch_first=True)
            self._mha_batch_first = True
        except TypeError:
            self.mha = nn.MultiheadAttention(vit_feat_dim, num_heads=num_heads)
            self._mha_batch_first = False
        # 与 ViT Block 一致的 FFN（Mlp），含 drop（无 drop_path 时等价于 0）
        self.mlp_v = Mlp(in_features=vit_feat_dim, hidden_features=int(vit_feat_dim * self.mlp_ratio), act_layer=nn.GELU, drop=self.drop)
        self.mlp_f = Mlp(in_features=vit_feat_dim, hidden_features=int(vit_feat_dim * self.mlp_ratio), act_layer=nn.GELU, drop=self.drop)
        self.drop_path_v = DropPath(self.drop_path_rate) if self.drop_path_rate > 0.0 else nn.Identity()
        self.drop_path_f = DropPath(self.drop_path_rate) if self.drop_path_rate > 0.0 else nn.Identity()

        # helper: unified MHA forward regardless of batch_first support
        def _mha_forward(q, k, v):
            if self._mha_batch_first:
                out, _ = self.mha(q, k, v)
                return out
            else:
                out, _ = self.mha(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
                return out.transpose(0, 1)
        self._mha_forward = _mha_forward

        # After concatenation at classifier, fuse back to ViT embed dim
        fusion_dim = vit_feat_dim + lsnet_feat_dim
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, max(128, fusion_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(128, fusion_dim // 2), vit_feat_dim)
        )
        # Classification will use original ViT head on fused features

    @torch.no_grad()
    def _to_gray(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x in (N, 3, H, W), convert to grayscale using luminance weights, then repeat to 3 channels for LSNet
        if x.size(1) == 1:
            g = x
        else:
            r, gch, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            g = 0.299 * r + 0.587 * gch + 0.114 * b
        return g

    def _freq_tokens_after_conv3(self, x: torch.Tensor):
        if self.rgb_dwt:
            ll, lh, hl, hh = self.dwt(x)
            x_hf = torch.cat([lh, hl, hh], dim=1)
            x_hf = self.rgb_reduce(x_hf)
        else:
            g = self._to_gray(x)
            ll, lh, hl, hh = self.dwt(g)
            x_hf = torch.cat([lh, hl, hh], dim=1)
        y1 = self.c1(x_hf)
        y2 = self.c2(y1)
        y3 = self.c3(y2)
        N, C, H, W = y3.shape
        tokens_f = y3.flatten(2).transpose(1, 2)
        return tokens_f, (y1, y2, y3)

    def forward(self, x: torch.Tensor, if_boundaries: torch.Tensor = None, return_xray: bool = False):
        # Prepare frequency branch up to conv3 to get tokens
        tokens_f, (y1, y2, y3) = self._freq_tokens_after_conv3(x)

        # Define cross-update fn for ViT after block-7
        def cross_update_vit(tokens_v: torch.Tensor) -> torch.Tensor:
            # tokens_v: (N, L, C_v). Align freq tokens to vit dim and same batch length
            N, L_v, C_v = tokens_v.shape
            Nf, L_f, C_f = tokens_f.shape
            assert N == Nf, 'Batch mismatch between ViT and Freq tokens'
            # project freq channels (256) -> vit dim C_v
            f_proj = self.proj_freq2vit(tokens_f)  # (N, L_f, C_v)
            # 分支A（pre-norm）：Q=LN(T_vit)，K/V=LN(T_freq_proj)
            q = self.norm_v_attn(tokens_v)
            kv = self.norm_f_attn(f_proj)
            attn_v = self._mha_forward(q, kv, kv)
            tokens_v = tokens_v + self.cross_alpha_v * self.drop_path_v(attn_v)
            tokens_v = tokens_v + self.cross_alpha_v * self.drop_path_v(self.mlp_v(self.norm_v_mlp(tokens_v)))
            return tokens_v

        # Run ViT with injection after block 7
        cls_feat, moe_loss, tokens, tokens8, fused_768 = self.vit.forward_features(x, cross_update_fn=cross_update_vit)

        # Branch B: update freq tokens with ViT tokens (use tokens8 if present else tokens)
        with torch.set_grad_enabled(self.training):
            tokens_v_src = tokens8 if tokens8 is not None else tokens
            # ensure (N, L, C)
            if tokens_v_src.dim() == 2:
                tokens_v_src = tokens_v_src.unsqueeze(1)
            f_proj = self.proj_freq2vit(tokens_f)
            # 分支B（pre-norm）：Q=LN(T_freq_proj)，K/V=LN(T_vit_src)
            qf = self.norm_f_attn(f_proj)
            kvv = self.norm_v_attn(tokens_v_src)
            attn_f = self._mha_forward(qf, kvv, kvv)
            tokens_f_updated = f_proj + self.cross_alpha_f * self.drop_path_f(attn_f)
            tokens_f_updated = tokens_f_updated + self.cross_alpha_f * self.drop_path_f(self.mlp_f(self.norm_f_mlp(tokens_f_updated)))
            # 将 tokens 从 ViT 维度映射回 256 通道，再还原为 (N, 256, H, W)
            tokens_f_back = self.proj_vit2freq_tokens(tokens_f_updated)  # (N, Lf, 256)
            N, Lf, Cf = tokens_f_back.shape
            H, W = y3.shape[-2], y3.shape[-1]
            y3_updated = tokens_f_back.transpose(1, 2).reshape(N, Cf, H, W)
            # continue conv4 and global pool → 384-d
            y4 = self.c4(y3_updated)
            ls_feat = F.adaptive_avg_pool2d(y4, 1).flatten(1)
            ls_feat = self.ls_project(ls_feat)

        # Final fusion for classification
        fused = torch.cat([cls_feat, ls_feat], dim=1)
        fused = self.fuse(fused)
        logits = self.vit.head(fused)

        if return_xray and getattr(self.vit, 'xray_postprocess', None) is not None and if_boundaries is not None:
            try:
                # use vit internal tokens to compute xray
                if tokens8 is not None:
                    patch_tokens = tokens8[:, 1:, :].permute(0, 2, 1)
                else:
                    patch_tokens = tokens[:, 1:, :].permute(0, 2, 1)
                xray_pred = self.vit.xray_postprocess(patch_tokens, if_boundaries)
            except Exception:
                xray_pred = None
            return logits, moe_loss, xray_pred

        return logits, moe_loss




