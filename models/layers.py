"""
Code from: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/lifelong/models/modules/rgb_modules.py 

This file contains all neural modules related to encoding the spatial
information of obs_t, i.e., the abstracted knowledge of the current visual
input conditioned on the language.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from .utils import build_grid


class SoftPositionEmbed(nn.Module):
    """Soft PE mapping normalized coords to feature maps."""

    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.dense = nn.Linear(in_features=4, out_features=hidden_size)
        self.register_buffer('grid', build_grid(resolution))  # [1, H, W, 4]

    def forward(self, inputs):
        """inputs: [B, C, H, W]."""
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2).contiguous()
        return inputs + emb_proj

class GridProjection(nn.Module):
    """Project grid to feature map size."""

    def __init__(self, visual_resolution, hidden_dim, out_features):
        super().__init__()
        self.pos_emb = SoftPositionEmbed(hidden_dim, visual_resolution)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )
    def forward(self, inputs):
        """inputs: [B, C, H, W]."""
        inputs = self.pos_emb(inputs) # [B, C, H, W]
        inputs = torch.flatten(inputs, start_dim=2, end_dim=3) # [B, C, H*W]
        inputs = inputs.permute(0, 2, 1).contiguous() # [B, H*W, C]
        return self.projection(inputs) # [B, H*W, out_features]

class RelativePosition(nn.Module):
    def __init__(self, num_embeds, max_relative_position):
        super().__init__()
        self.num_embeds = num_embeds
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_embeds))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings

class InverseCrossAttentionMH(nn.Module):
    """Cross-attention module with inverted-attention and no dropout
    as presented in: https://openreview.net/pdf?id=m9s6rnYWqm"""
    def __init__(
            self, 
            d_model, 
            num_heads, 
        ):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_scale = d_model ** -0.5
        
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.proj_o = nn.Linear(d_model, d_model, bias=False)
    
    
    def forward(self, q, k, v, attn_mask=None, return_seg=False):
        """
        q: B x T x D
        k: B x S x D
        v: B x S x D
        attn_mask: T x S
        return: B x T x F
        """
        B, T, _, = q.shape
        _, S, _ = k.shape
        seg_mask = None 

        # Project and split to different heads
        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2) # [B, num_heads, T, D]
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2) # [B, num_heads, S, D]
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2) # [B, num_heads, S, D]     
        
        q = q * self.attn_scale 
        attn = torch.matmul(q, k.transpose(-1, -2)) # [B, num_heads, T, S]
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        # Inverted-attention + Renoemalization
        attn = F.softmax(attn, dim=-2) # normalize along query; -1 for key
        attn = attn / torch.sum(attn, dim=-1, keepdim=True)
        if return_seg:
            seg_mask = attn.detach().clone().permute(0, 2, 1)
        # Concat
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        output = self.proj_o(output)

        return output, seg_mask
    

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class PSBBlock(nn.Module):
    def __init__(
            self,
            embed_dim=192,
            ffn_dim=512,
            inverse_nh=1,
            time_nh=4,
            obj_nh=4,
            num_slots=8,
            get_mask=False,
        ):
        super().__init__()

        ## Inverse Cross-Attention
        self.inverse_cross_attn = InverseCrossAttentionMH(
            d_model=embed_dim,
            num_heads=inverse_nh,
        )
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.pos_emb = RelativePosition(embed_dim, 32)
        self.num_slots = num_slots
        self.init_slots = nn.Parameter(
            nn.init.normal_(torch.empty(1, 1, self.num_slots, embed_dim)))
        self.get_mask = get_mask
        
        ## Time-Axis Self-Attention
        self.layer_time = nn.LayerNorm(embed_dim)
        self.time_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=time_nh,
            )

        ## Object-Axis Self-Attention
        self.layer_obj = nn.LayerNorm(embed_dim)
        self.obj_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=obj_nh,
            )

        ## Final Projection
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, input):
        x, prev_slots, mask = input["features"], input["prev_slots"], input["mask"]
        B, T, L, D = x.shape
        N = self.num_slots
        # B x T x N x D
        if prev_slots is None:
            prev_slots = self.init_slots.repeat(B, T, 1, 1)
        input = x

        # Inverse Cross-Attention
        slots = self.ln_q(prev_slots)
        x = self.ln_kv(x)
        slots = einops.rearrange(slots, 'b t n d -> (b t) n d', b=B, n=N, t=T)
        x = einops.rearrange(x, 'b t n d -> (b t) n d', b=B, t=T)
        slots, attn_mask = self.inverse_cross_attn(slots, x, x, return_seg=self.get_mask) # B x T x N x D

        # Time-Axis Self-Attention
        # slots = einops.rearrange(slots, 'b t n d -> (b t) n d', b=B, n=N, t=T)
        slots = self.layer_time(slots)
        slots, _ = self.time_attn(slots, slots, slots)

        # Object-Axis Self-Attention
        slots = einops.rearrange(slots, '(b t) n d -> (b n) t d', b=B, n=N, t=T)
        slots = self.layer_obj(slots)
        slots, _ = self.obj_attn(slots, slots, slots)

        # Final Projection
        slots = slots.view(B, T, N, D)
        slots = self.proj(slots) + prev_slots
        out_dict = {
            "features": input,
            "prev_slots": prev_slots,
            "mask": None
        }
        if self.get_mask:
            out_dict["mask"] = attn_mask
        return out_dict
