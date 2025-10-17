"""
Adapted from: https://github.com/openai/CLIP/blob/main/clip/clip.py
"""
from collections import OrderedDict
from typing import Tuple, Union

import hashlib
import os
import urllib
import warnings
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn


_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}
_PT_NAME = {
    "RN50": "RN50.pt",
    "RN101": "RN101.pt",
    "RN50x4": "RN50x4.pt",
    "RN50x16": "RN50x16.pt",
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
}

def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def available_models():
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

# =============================
# ===== LVPRUNING MODS: utils =====
import math
from typing import Optional, List

class _LVDebugRecorder:
    def __init__(self):
        self.on = False
        self.rec = {}          # layer_i -> dict(x, policy_BL, before_params)
        self.param_names = []  # filled once
    def reset(self):
        self.rec.clear()
    def enable(self, on=True):
        self.on = on

LV_DEBUG = _LVDebugRecorder()

def gumbel_softmax_sample(logits: torch.Tensor, tau: float, eps: float = 1e-6):
    # 1) 采样在 fp32，clamp 防 log(0)
    U = torch.rand_like(logits, dtype=torch.float32).clamp_(eps, 1.0 - eps)
    g = -torch.log(-torch.log(U))  # fp32

    # 2) logits 升 fp32，并做数值中心化（防溢出）
    logits32 = logits.to(torch.float32)
    logits32 = logits32 - logits32.max(dim=-1, keepdim=True).values  # LSE trick

    # 3) tau 设下界（太小极易 NaN）；你可以先从 0.5 起步
    tau = max(float(tau), 5e-2)

    y = (logits32 + g) / tau

    # 4) softmax 在 fp32，最后再 cast 回原 dtype
    probs = torch.softmax(y, dim=-1).to(logits.dtype)

    # 5) 一次性检查（如果还有问题，直接抛出更明确的定位）
    if not torch.isfinite(probs).all():
        mn = torch.nan_to_num(probs.detach().float(), nan=0., posinf=1e9, neginf=-1e9).min().item()
        mx = torch.nan_to_num(probs.detach().float(), nan=0., posinf=1e9, neginf=-1e9).max().item()
        raise RuntimeError(f"[Gumbel] probs non-finite: min={mn:.3e} max={mx:.3e}")
    return probs

def straight_through_hard_sample(probs: torch.Tensor):
    # probs: (..., 2)  -> hard one-hot with STE
    idx = probs.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
    return hard + (probs - probs.detach())

class LVDecisionModule(torch.nn.Module):
    """
    language-guided pruning head:
      Q: vision tokens at this layer (exclude [CLS])
      K,V: text tokens (full-length)
    logit -> 2-class (drop/keep)
    """
    def __init__(self, vis_dim: int, txt_dim: int, n_head: int, inner_dim: int=None):
        super().__init__()
        self.q_proj = nn.Linear(vis_dim, inner_dim)   # Q ← 视觉 768
        self.k_proj = nn.Linear(txt_dim, inner_dim)   # K ← 文本 512
        self.v_proj = nn.Linear(txt_dim, inner_dim)   # V ← 文本 512
        self.n_head = n_head
        self.inner_dim = inner_dim or vis_dim
        assert self.inner_dim % n_head == 0, "inner_dim must be divisible by n_head"
        self.d_head = self.inner_dim // n_head
        self.out_mlp = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, inner_dim),
            QuickGELU(),
            torch.nn.Linear(inner_dim, 2)  # [drop, keep]
        )

    def forward(self, vis_tokens: torch.Tensor, text_tokens: torch.Tensor):
        """
        vis_tokens: [B, Nv, C] (exclude cls)
        text_tokens: [B, Lt, C]
        return: logits [B, Nv, 2]
        """
        B, Nv, C = vis_tokens.shape
        Lt = text_tokens.shape[1]

        q = self.q_proj(vis_tokens).view(B, Nv, self.n_head, self.d_head).transpose(1, 2)    # [B,H,Nv,Dh]
        k = self.k_proj(text_tokens).view(B, Lt, self.n_head, self.d_head).transpose(1, 2)   # [B,H,Lt,Dh]
        v = self.v_proj(text_tokens).view(B, Lt, self.n_head, self.d_head).transpose(1, 2)   # [B,H,Lt,Dh]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)                            # [B,H,Nv,Lt]
        attn = torch.softmax(attn, dim=-1)
        ctx  = attn @ v                                                                      # [B,H,Nv,Dh]
        ctx  = ctx.transpose(1, 2).contiguous().view(B, Nv, self.inner_dim)                  # [B,Nv,C]

        fused = ctx + vis_tokens[..., :self.inner_dim]                                       # residual
        logits = self.out_mlp(fused)                                                         # [B,Nv,2]

        # === DPP-ATTN: 从现成注意力得到“文本条件相关性” q_i
        rel_BN = attn.mean(dim=-1).mean(dim=1)
        return logits, rel_BN  # [B,Nv,2], [B,Nv]

class LVPruneController(torch.nn.Module):
    """
    Manage multi-layer pruning: accumulate policy and provide L_ratio.
    """
    def __init__(self, vis_dim: int, txt_dim: int, n_head: int, prune_layers: List[int], keep_ratios: List[float], tau: float = 1.0):
        assert len(prune_layers) == len(keep_ratios)
        super().__init__()
        self.layers = prune_layers
        self.keep_ratios = keep_ratios
        self.tau = tau

        inner_dim = vis_dim
        assert inner_dim % n_head == 0

        self.heads = torch.nn.ModuleList([LVDecisionModule(vis_dim=vis_dim, txt_dim=txt_dim, n_head=n_head, inner_dim=inner_dim) for _ in prune_layers])
        # self.heads.half()

        # runtime states
        self.layer2policy = {}   # layer_idx -> [B, L] (including CLS at index 0)
        self.cum_policy   = None # [B, L] applied to *subsequent* layers
        self.loss_ratio   = 0.0

        # ===== LVPRUNING DEBUG =====
        self.debug = True

        # === DPP-ATTN 正则（轻量）：只用一层线性做核特征
        self.loss_dpp     = 0.0
        self.dpp_eps      = 1e-5
        dpp_dim = 256         # 可按需降到 256/384 节省算力
        self.dpp_vproj    = nn.Linear(vis_dim, dpp_dim, bias=False)
        self.dpp_quality_power    = 1.0
        self.dpp_keepprob_power   = 1.0

    def step(self, layer_i: int, x_LND: torch.Tensor, text_hidden_BLC: torch.Tensor, training: bool):
        """
        x_LND: [L, B, C] (vision tokens with CLS at 0)
        text_hidden_BLC: [B, Lt, C] (projected by CLIP text head, see encode_text(return_hidden=True))
        returns current cumulative policy [L,B] for applying in later blocks (LND broadcast)
        """
        if layer_i not in self.layers:
            return self.cum_policy

        B = x_LND.size(1)
        L = x_LND.size(0)

        # split CLS + patches
        vis_all_NLC = x_LND.permute(1,0,2)   # [B,L,C]
        vis_cls = vis_all_NLC[:, :1, :]
        vis_tok = vis_all_NLC[:, 1:, :]

        head = self.heads[self.layers.index(layer_i)]
        with torch.cuda.amp.autocast(enabled=False):
            logits, rel_BN = head(vis_tok.float(), text_hidden_BLC.float())
        if not torch.isfinite(logits).all():
            raise RuntimeError("[LV][HEAD logits] non-finite")

        if training:
            probs = gumbel_softmax_sample(logits, self.tau)  # [B,L-1,2]
            if not torch.isfinite(probs).all():
                # 直接报出第几层出问题
                raise RuntimeError(f"[LV][Gumbel probs] non-finite at layer {layer_i}")
            hard  = straight_through_hard_sample(probs)      # STE
            if not torch.isfinite(hard).all():
                raise RuntimeError(f"[LV][hard_st] non-finite at layer {layer_i}")
            keep_prob = probs[..., 1]
            keep_bin  = hard[..., 1]                         # {0,1}

            # ratio regularizer (match avg keep to target p_l)
            target = self.keep_ratios[self.layers.index(layer_i)]
            keep_avg = keep_prob.mean()
            self.loss_ratio = self.loss_ratio + (keep_avg - target) ** 2

            curr = torch.cat([torch.ones(B,1, device=x_LND.device, dtype=keep_bin.dtype), keep_bin], dim=1)  # add CLS keep=1

            if rel_BN is not None:
                vis_all_NLC = x_LND.permute(1, 0, 2)
                vis_tok_BNC = vis_all_NLC[:, 1:, :]
                keep_prob_BN = probs[..., 1]
                dpp_loss = self._dpp_from_attn(vis_tok_BNC, keep_prob_BN, rel_BN)
                self.loss_dpp = self.loss_dpp + dpp_loss
        else:
            # eval: top-k by keep-logit
            k = max(1, int(round(self.keep_ratios[self.layers.index(layer_i)] * (L-1))))
            scores = logits[..., 1]                      # [B,L-1]
            topk_idx = scores.topk(k, dim=-1).indices
            curr = torch.zeros(B, L-1, device=x_LND.device, dtype=x_LND.dtype)
            curr.scatter_(1, topk_idx, 1.0)
            curr = torch.cat([torch.ones(B,1, device=x_LND.device, dtype=curr.dtype), curr], dim=1)

        # accumulate with previous policy (progressive pruning)
        if self.cum_policy is None:
            self.cum_policy = curr  # [B,L]
        else:
            self.cum_policy = self.cum_policy * curr
            
        self.debug = False
        # ===== LVPRUNING DEBUG: retain grad & snapshot =====
        if self.debug and LV_DEBUG.on and x_LND.requires_grad:
            # 记录进入本层前的输入 x（作为本层“被 mask 的K/V来源”形状对齐点）
            x_LND.retain_grad()
            # policy: [B,L]，保留给统计用
            policy_BL = self.cum_policy.detach().clone()
            # 首次：记录头参数名（用于打印有序结果）
            if not LV_DEBUG.param_names:
                for s, head in zip(self.layers, self.heads):
                    for name, p in head.named_parameters():
                        LV_DEBUG.param_names.append(f"head{self.layers.index(s)}.{name}")
                        break  # 只需要占位一次即可
            # 保存本层条目
            LV_DEBUG.rec[layer_i] = {
                "x": x_LND,                      # [L,B,C], 已 retain_grad
                "policy_BL": policy_BL,          # [B,L]
                "param_before": [p.detach().clone() for p in self.heads[self.layers.index(layer_i)].parameters()]
            }

        # cache per-layer (optional diagnostics)
        self.layer2policy[layer_i] = self.cum_policy.detach()

        self._last_policy_BL = self.cum_policy.detach().clone()
        return self.cum_policy  # [B,L]
    
    @staticmethod
    def _safe_logdet_psd(A: torch.Tensor, eps: float) -> torch.Tensor:
        # A: [B,N,N]，假设 PSD
        B, N, _ = A.shape
        I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
        A = A + eps * I
        L = torch.linalg.cholesky(A.to(torch.float32))
        logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)  # [B]
        return logdet.to(A.dtype)
    
    def _dpp_from_attn(self,
                    vis_tok_BNC: torch.Tensor,   # [B, N, C]  本层视觉 tokens（不含 CLS）
                    keep_prob_BN: torch.Tensor,  # [B, N]     剪枝头 “保留” 概率
                    rel_BN: torch.Tensor         # [B, N]     cross-attn 平均得到的相关性
                    ) -> torch.Tensor:
        """
        依据 DPP-Attention：L = (Φ·Diag(q)) (Φ·Diag(q))^T，q_i ~ sqrt(rel_i^γq * kp_i^γk)
        返回：标量 dpp_loss = - mean_B logdet(•)
        """
        # ---------- 质量：相关性 rel ----------
        eps = 1e-12
        rel = rel_BN.clamp_min(0.0)                                # 非负
        rel = rel / (rel.mean(dim=1, keepdim=True) + eps)          # 批内尺度不变
        if getattr(self, "dpp_quality_power", 1.0) != 1.0:
            rel = rel.pow(self.dpp_quality_power)                  # γ_q

        # ---------- 保留概率：keep_prob ----------
        kp = keep_prob_BN.clamp(0.0, 1.0)
        if getattr(self, "dpp_keepprob_power", 1.0) != 1.0:
            kp = kp.pow(self.dpp_keepprob_power)                   # γ_k

        # ---------- 软权重：进入核前取 sqrt ----------
        w = rel * kp                                               # [B, N]
        w = (w / (w.mean(dim=1, keepdim=True) + eps)).clamp_min(1e-6)
        w = w.sqrt().unsqueeze(-1)                                 # [B, N, 1]

        # ---------- 多样性特征：线性投影（或 Identity） + L2 归一 ----------
        with torch.cuda.amp.autocast(enabled=False):               # DPP 分支统一 fp32 更稳
            phi = self.dpp_vproj(vis_tok_BNC.to(torch.float32))    # [B, N, d]
            phi = torch.nn.functional.normalize(phi, dim=-1)
            w32 = w.to(torch.float32)

            # ---------- Gram 核 ----------
            G = (phi * w32) @ (phi * w32).transpose(1, 2)          # [B, N, N]  PSD, fp32

            # ---------- 可选稳健归一化（默认开启） ----------
            use_norm = getattr(self, "dpp_use_norm", True)
            tau = float(getattr(self, "dpp_tau", 1e-2))  # ← 加：温度，默认 1.0

            if use_norm:
                d = max(1, int(phi.size(-1)))
                n = max(1, int(G.size(-1)))
                # 先按 d、N 归一，再按 τ 缩放：G ← G / (d * n * τ)
                G = G / (float(d) * float(n) * max(tau, 1e-6))   # ← 改：加入 tau
                A = G + torch.eye(n, device=G.device, dtype=G.dtype).unsqueeze(0)
                logdet = self._safe_logdet_psd(A, getattr(self, "dpp_eps", 1e-5))  # [B]
            else:
                # 即使不用归一化，也可用 τ：A = I + G/τ
                n = max(1, int(G.size(-1)))
                A = (G / max(tau, 1e-6)) + torch.eye(n, device=G.device, dtype=G.dtype).unsqueeze(0)
                logdet = self._safe_logdet_psd(A, getattr(self, "dpp_eps", 1e-5))

        return -logdet.mean()

    
    # ===== LVPRUNING DEBUG: helpers =====
    def debug_enable(self, on=True):
        self.debug = on
        LV_DEBUG.enable(on)
        if on:
            LV_DEBUG.reset()

    @torch.no_grad()
    def debug_report(self):
        if not (self.debug and LV_DEBUG.on):
            return {}
        report = {}
        for li, entry in LV_DEBUG.rec.items():
            x = entry["x"]                # [L,B,C], has .grad
            pol = entry["policy_BL"]      # [B,L]
            if x.grad is None:
                # 说明 backward 没到或被关闭
                kept_grad_mean = drop_grad_mean = 0.0
                kept_cnt = drop_cnt = 0
            else:
                # 按 token 维统计梯度范数
                g = x.grad                 # [L,B,C]
                g_norm = g.pow(2).sum(-1).sqrt()   # [L,B]
                g_norm_BL = g_norm.permute(1,0)    # [B,L]
                kept_mask = (pol > 0.5)
                drop_mask = ~kept_mask
                # 避免除 0
                kept_cnt = kept_mask.sum().item()
                drop_cnt = drop_mask.sum().item()
                kept_grad_mean = (g_norm_BL[kept_mask].mean().item() if kept_cnt > 0 else 0.0)
                drop_grad_mean = (g_norm_BL[drop_mask].mean().item() if drop_cnt > 0 else 0.0)

            # 头参数梯度范数
            head = self.heads[self.layers.index(li)]
            p_grad_norm = 0.0
            has_grad = False
            for p in head.parameters():
                if p.grad is not None:
                    has_grad = True
                    p_grad_norm += p.grad.data.float().norm().item()
            # 参数是否真的更新（需要在 optimizer.step() 后再调这个函数）
            changed = []
            for p_before, p_after in zip(entry["param_before"], head.parameters()):
                changed.append(float((p_after.detach() - p_before).abs().max().item()))
            max_update = max(changed) if changed else 0.0

            report[li] = {
                "kept_cnt": kept_cnt, "drop_cnt": drop_cnt,
                "kept_grad_mean": kept_grad_mean,
                "drop_grad_mean": drop_grad_mean,
                "head_grad_norm_sum": (p_grad_norm if has_grad else 0.0),
                "param_max_update": max_update,
            }
        return report

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, policy_LB: torch.Tensor = None):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))   # LND

        L, B, C = x.shape
        H = self.attn.num_heads

        # ===== 训练期 + 有策略：稳定版“加性掩码（STE）” =====
        if self.training and (policy_LB is not None):
            L, B, C = x.shape
            H = self.attn.num_heads

            # 1) policy: [L,B] -> [B,L]，避免in-place写，CLS强制保留
            key_keep_BL = policy_LB.transpose(0, 1).contiguous()     # [B,L] 0/1(≈)
            if key_keep_BL.size(1) > 0:
                key_keep_BL = key_keep_BL.clone()                    # 新叶子张量，安全
                key_keep_BL[:, 0] = 1.0                              # CLS 永远 keep

            # 2) “前向强惩罚 / 反向小梯度”的可微大负掩码（防回传NaN/爆炸）
            #    drop = 1 - keep；前向：-M_big * drop；反向：-m_grad * d(drop)
            M_big  = 1e4   # 前向大负（≈ -inf）
            m_grad = 10.0  # 反向梯度系数（建议 5~10）
            drop = (1.0 - key_keep_BL)                               # [B,L]，带梯度
            mask_1D = (-M_big) * drop.detach() + (-m_grad) * (drop - drop.detach())  # [B,L]

            # 3) 扩到 [B*H, Lk]→[B*H, Lq, Lk]，在 fp32 构造更稳
            mask_k = mask_1D.unsqueeze(1).repeat(1, H, 1).reshape(B * H, L)          # [B*H, L]
            add3d  = mask_k.to(torch.float32).unsqueeze(1).repeat(1, L, 1)           # [B*H, L, L]

            # 4) 合并结构性mask（若有），统一 fp32
            if attn_mask_ is not None:
                base = attn_mask_.to(dtype=torch.float32, device=x.device)           # [L,L]
                base = base.unsqueeze(0).unsqueeze(0).expand(B, H, L, L).reshape(B * H, L, L)
                add3d = add3d + base

            # 5) 安全兜底 & dtype 对齐（MHA要求：bool 或与 query 同dtype）
            add3d = torch.nan_to_num(add3d, nan=-1e4, posinf=-1e4, neginf=-1e4)
            add3d = add3d.to(dtype=x.dtype)

            # 6) 调原生 MHA（它会把float mask当作logits偏置加到softmax前），更稳更快
            ctrl = getattr(self, "_lv_controller_ref", None)
            if ctrl is not None:
                ctrl._attn_fp32 = True
            return self.attn(x, x, x, need_weights=False, attn_mask=add3d)[0]

        else:
            # 默认：不屏蔽任何 key
            key_padding_mask = None
            if policy_LB is not None:
                # policy_LB: [L, B] -> [B, L]；True 表示“屏蔽/丢弃”
                key_keep_BL = policy_LB.transpose(0, 1).contiguous()    # [B, L], 1=keep, 0=drop
                key_padding_mask = (key_keep_BL < 0.5)                  # bool [B, L], True=mask(drop)


            attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_, key_padding_mask=key_padding_mask)[0]

    def forward(self, x_tuple: tuple):
        # x_tuple can be (x, video_frame) or (x, video_frame, policy_LB)
        if len(x_tuple) == 2:
            x, video_frame = x_tuple
            policy_LB = None
        else:
            x, video_frame, policy_LB = x_tuple  # policy over tokens for each sample

        x = x + self.attention(self.ln_1(x), policy_LB=policy_LB)  # masked attention if provided
        x = x + self.mlp(self.ln_2(x))
        return (x, video_frame, policy_LB)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, video_frame=-1, policy_LB: Optional[torch.Tensor] = None):
        out = (x, video_frame, policy_LB)
        for blk in self.resblocks:
            out = blk(out)
        x, _, policy_LB = out
        return x

class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 linear_patch: str = '2d',):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # For 3D
        assert linear_patch in ['2d', '3d']
        self.linear_patch = linear_patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size), padding=(1, 0, 0), bias=False)

    def forward(self, x: torch.Tensor, video_frame=-1, lvprune_cfg: Optional[dict] = None):

        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            x_3d = self.conv2(x_3d)     # shape = [*, width, frame, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)      # shape = [*, frame, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous() # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if lvprune_cfg is None:
            x = self.transformer(x, video_frame=video_frame)
        else:
            ctrl: LVPruneController = lvprune_cfg["controller"]
            ctrl.layer2policy.clear()   # ✅ 清空“上一轮的快照”，避免挂旧图
            ctrl.cum_policy = None      # ✅ 重新从全保留开始累计（当前这轮会重新计算）
            ctrl.loss_ratio = 0.0       # ✅ 本轮重新累计比例正则
            text_hidden = lvprune_cfg["text_hidden"]                 # [B,Lt,C]
            prune_set = set(ctrl.layers)

            policy_LB = None  # [L,B]
            for li, blk in enumerate(self.transformer.resblocks):
                if li in prune_set:
                    policy_B_L = ctrl.step(li, x, text_hidden, training=self.training)  # [B,L]
                    policy_LB = policy_B_L.transpose(0, 1).contiguous()                 # [L,B]
                x, _, policy_LB = blk((x, video_frame, policy_LB))
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Move the three lines below to `encode_image` for entire hidden sequence
        # x = self.ln_post(x[:, 0, :])
        # if self.proj is not None:
        #     x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # vision linear of patch
                 linear_patch: str = '2d',
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch=linear_patch
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @staticmethod
    def get_config(pretrained_clip_name="ViT-B/32"):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViT-B-32.pt")
        if pretrained_clip_name in _MODELS and pretrained_clip_name in _PT_NAME:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[pretrained_clip_name])

        if pretrained_clip_name in ["ViT-B/32", "ViT-B/16"] and os.path.exists(model_path):
            pass
        else:
            if pretrained_clip_name in _MODELS:
                model_path = _download(_MODELS[pretrained_clip_name])
            elif os.path.isfile(pretrained_clip_name):
                model_path = pretrained_clip_name
            else:
                raise RuntimeError(f"Model {pretrained_clip_name} not found; available models = {available_models()}")

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        return state_dict

    def build_attention_mask(self, context_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_hidden=False, video_frame=-1, lvprune_cfg: Optional[dict] = None):
        hidden = self.visual(image.type(self.dtype), video_frame=video_frame, lvprune_cfg=lvprune_cfg)
        hidden = self.visual.ln_post(hidden) @ self.visual.proj

        x = hidden[:, 0, :]

        if return_hidden:
            return x, hidden

        return x

    def encode_text(self, text, return_hidden=False):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = x + pos_emd
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]

        if return_hidden:
            return x, hidden

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
