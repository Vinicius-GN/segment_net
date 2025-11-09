import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple


class _IdentityNorm(nn.Module):
    def forward(self, x):
        return x


def _make_norm(kind: str, num_channels: int) -> nn.Module:
    if kind == "batch_norm":
        return nn.BatchNorm2d(num_channels)
    if kind == "group_norm":
        g = 32
        while g > 1 and (num_channels % g != 0):
            g -= 1
        if g <= 1:
            return nn.GroupNorm(1, num_channels)
        return nn.GroupNorm(g, num_channels)
    return _IdentityNorm()


def _ensure_list_features(x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _pick_lowest_res(feats: List[torch.Tensor]) -> torch.Tensor:
    return min(feats, key=lambda t: int(t.shape[-2]) * int(t.shape[-1]))


def _pick_highest_res(feats: List[torch.Tensor]) -> torch.Tensor:
    return max(feats, key=lambda t: int(t.shape[-2]) * int(t.shape[-1]))


class _FFN(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(d_model, dim_feedforward)
        self.lin2 = nn.Linear(dim_feedforward, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, x):
        y = self.lin2(self.dropout(self.act(self.lin1(x))))
        return y


class _MaskedXAttn(nn.Module):
    """
    Multi-head masked cross-attention between queries and pixel tokens.
    Mask is (B, Q, HW) with True=keep, False=mask-out.
    """
    def __init__(self, d_model: int, nheads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nheads = nheads
        self.dk = d_model // nheads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, q: torch.Tensor, mem: torch.Tensor, keep_mask: torch.Tensor):
        # q: [B,Q,C], mem: [B,HW,C], keep_mask: [B,Q,HW] (bool)
        B, Q, C = q.shape
        _, HW, _ = mem.shape
        qh = self.q_proj(q).view(B, Q, self.nheads, self.dk).transpose(1, 2)        
        kh = self.k_proj(mem).view(B, HW, self.nheads, self.dk).transpose(1, 2)     
        vh = self.v_proj(mem).view(B, HW, self.nheads, self.dk).transpose(1, 2)     
        attn_logits = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.dk)  

        if keep_mask is not None:
            m = keep_mask.unsqueeze(1)                                              
            attn_logits = attn_logits.masked_fill(~m, float("-inf"))

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, vh)                                                
        out = out.transpose(1, 2).contiguous().view(B, Q, C)
        out = self.out_proj(out)
        return out


class _DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nheads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)
        self.cross_attn = _MaskedXAttn(d_model, nheads, dropout)
        self.ffn = _FFN(d_model, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, q: torch.Tensor, mem: torch.Tensor, keep_mask: torch.Tensor):
        # self-attn
        y, _ = self.self_attn(q, q, q, need_weights=False)
        q = q + self.dropout(y)
        q = self.norm1(q)
        # masked cross-attn
        y = self.cross_attn(q, mem, keep_mask)
        q = q + self.dropout(y)
        q = self.norm2(q)
        # ffn
        y = self.ffn(q)
        q = q + self.dropout(y)
        q = self.norm3(q)
        return q


class Mask2FormerHead(nn.Module):
    """
    Mask2Former-style decode head with masked cross-attention and mask classification fusion.
    Returns semantic logits [B, C, H, W] for compatibility with your pipeline.
    Exposes last query/class/mask tensors via self.last for advanced training use.
    """

    def __init__(self, config, num_classes: int, num_feature_layers: int):
        super().__init__()
        self.config = config
        self.num_classes = int(num_classes)
        self.num_feature_layers = int(num_feature_layers)

        self.image_size = list(reversed(self.config.get("image").get("image_size")))
        self.opt_latency = bool(self.config.get("head").get("opt_latency"))
        self.dropout_prob = float(self.config.get("head").get("dropout", 0.0))
        self.norm2d = (self.config.get("model").get("norm_fn") or "batch_norm")
        self.aggregate = (self.config.get("backbone").get("aggregate") or "").lower()

        self.num_queries = int(self.config.get("head").get("m2f_num_queries", 100))
        self.d_model = int(self.config.get("head").get("m2f_hidden_dim", 256))
        self.nheads = int(self.config.get("head").get("m2f_nheads", 8))
        self.ffn_dim = int(self.config.get("head").get("m2f_ffn_dim", 1024))
        self.num_layers = int(self.config.get("head").get("m2f_num_layers", 6))

        self.mask_dim = int(self.config.get("head").get("m2f_mask_dim", 256))
        self.mask_thresh = float(self.config.get("head").get("m2f_mask_threshold", 0.5))

        self._built = False
        self._in_chs = None

        self.proj_feats = None          
        self.mask_proj = None         
        self.decoder = None           
        self.query_embed = None        
        self.class_head = None        
        self.mask_embed = None        

        self.dropout = nn.Dropout(self.dropout_prob, inplace=False)
        self.last = {}

    def _maybe_build(self, feats: List[torch.Tensor]):
        in_chs = [int(t.shape[1]) for t in feats]
        if self._built and self._in_chs == tuple(in_chs):
            return
        device = feats[0].device
        dtype = feats[0].dtype

        self.proj_feats = nn.ModuleList(
            [nn.Conv2d(c, self.d_model, kernel_size=1, bias=False) for c in in_chs]
        ).to(device=device, dtype=dtype)

        # mask features from highest-res
        idx_hr = max(range(len(feats)), key=lambda i: int(feats[i].shape[-2]) * int(feats[i].shape[-1]))
        self._hr_index = idx_hr
        self.mask_proj = nn.Conv2d(in_chs[idx_hr], self.mask_dim, kernel_size=1, bias=False).to(device=device, dtype=dtype)

        # transformer decoder
        self.decoder = nn.ModuleList(
            [_DecoderLayer(self.d_model, self.nheads, self.ffn_dim, self.dropout_prob) for _ in range(self.num_layers)]
        ).to(device=device, dtype=dtype)

        # queries and heads
        self.query_embed = nn.Embedding(self.num_queries, self.d_model).to(device=device, dtype=dtype)
        self.class_head = nn.Linear(self.d_model, self.num_classes + 1).to(device=device, dtype=dtype)  # + no-object
        self.mask_embed = nn.Linear(self.d_model, self.mask_dim).to(device=device, dtype=dtype)

        self._in_chs = tuple(in_chs)
        self._built = True

    @staticmethod
    def _ensure_list(x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
        return _ensure_list_features(x)

    def _build_memory(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        tokens = []
        for i, f in enumerate(feats):
            m = self.proj_feats[i](f)                           
            B, C, Hi, Wi = m.shape
            tokens.append(m.flatten(2).transpose(1, 2))          
        mem = torch.cat(tokens, dim=1)                           
        # mask features from highest-res map
        fhr = feats[self._hr_index]
        mask_feats = self.mask_proj(fhr)                        
        return mem, mask_feats, fhr.shape[-2:]

    def _masked_keep(self, mask_logits: torch.Tensor) -> torch.Tensor:
        keep = (torch.sigmoid(mask_logits) >= self.mask_thresh)  
        B, Q, H, W = keep.shape
        return keep.view(B, Q, H * W)

    def _compute_masks(self, q: torch.Tensor, mask_feats: torch.Tensor) -> torch.Tensor:
        emb = self.mask_embed(q)                                 
        masks = torch.einsum("bqc,bchw->bqhw", emb, mask_feats)
        return masks

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
        feats = self._ensure_list(x)
        if self.aggregate == "concat" and len(feats) > 1:
            fcat = torch.cat(feats, dim=1)
            feats = [fcat]
        self._maybe_build(feats)

        mem, mask_feats, (Hm, Wm) = self._build_memory(feats)    
        B = mem.size(0)
        q = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) 

        keep = None
        mask_logits_per_layer = []
        for layer in self.decoder:
            q = layer(q, mem, keep)
            cur_masks = self._compute_masks(q, mask_feats)      
            mask_logits_per_layer.append(cur_masks)
            keep = self._masked_keep(cur_masks)                  

        class_logits = self.class_head(q)                         
        masks_final = mask_logits_per_layer[-1]                 
        class_prob = F.softmax(class_logits, dim=-1)[..., :-1]    

        mask_prob = torch.sigmoid(masks_final)                   
        sem = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)

        if self.opt_latency:
            eps = 1e-6
            sem = torch.clamp(sem, eps, 1 - eps)
            sem = torch.log(sem / (1 - sem))
            sem = F.interpolate(sem, size=self.image_size, mode="bilinear", align_corners=False)
        else:
            eps = 1e-6
            sem = torch.clamp(sem, eps, 1 - eps)
            sem = torch.log(sem / (1 - sem))
            sem = F.interpolate(sem, size=self.image_size, mode="bilinear", align_corners=False)

        self.last = {
            "query_feats": q,                    # [B,Q,C]
            "class_logits": class_logits,        # [B,Q,C+1]
            "mask_logits": masks_final,          # [B,Q,Hm,Wm]
            "all_mask_logits": mask_logits_per_layer,
        }
        return sem
