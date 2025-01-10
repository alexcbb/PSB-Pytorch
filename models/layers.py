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
from typing import Tuple
from .utils import init_parameters
from .attention import MemEffAttention

class MLP(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims,
        initial_layer_norm: bool = False,
        final_activation = False,
        residual: bool = False,
        weight_init: str = "default",
    ):
        super().__init__()
        self.residual = residual
        if residual:
            assert inp_dim == outp_dim

        layers = []
        if initial_layer_norm:
            layers.append(nn.LayerNorm(inp_dim))

        cur_dim = inp_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(cur_dim, dim))
            layers.append(nn.ReLU())
            cur_dim = dim

        layers.append(nn.Linear(cur_dim, outp_dim))
        if final_activation:
            if isinstance(final_activation, bool):
                final_activation = "relu"
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        init_parameters(self.layers, weight_init)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        outp = self.layers(inp)

        if self.residual:
            return inp + outp
        else:
            return outp

class SlotAttention(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        slot_dim: int,
        kvq_dim= None,
        hidden_dim = None,
        n_iters: int = 3,
        eps: float = 1e-8,
        use_gru: bool = True,
        use_mlp: bool = True,
    ):
        super().__init__()
        assert n_iters >= 1

        if kvq_dim is None:
            kvq_dim = slot_dim
        self.to_k = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_v = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_q = nn.Linear(slot_dim, kvq_dim, bias=False)

        if use_gru:
            self.gru = nn.GRUCell(input_size=kvq_dim, hidden_size=slot_dim)
        else:
            assert kvq_dim == slot_dim
            self.gru = None

        if hidden_dim is None:
            hidden_dim = 4 * slot_dim

        if use_mlp:
            self.mlp = MLP(
                slot_dim, slot_dim, [hidden_dim], initial_layer_norm=True, residual=True
            )
        else:
            self.mlp = None

        self.norm_features = nn.LayerNorm(inp_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.n_iters = n_iters
        self.eps = eps
        self.scale = kvq_dim**-0.5

    def step(
        self, slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one iteration of slot attention."""
        slots = self.norm_slots(slots)
        queries = self.to_q(slots)

        dots = torch.einsum("bsd, bfd -> bsf", queries, keys) * self.scale
        pre_norm_attn = torch.softmax(dots, dim=1)
        attn = pre_norm_attn + self.eps
        attn = attn / attn.sum(-1, keepdim=True)

        updates = torch.einsum("bsf, bfd -> bsd", attn, values)

        if self.gru:
            updated_slots = self.gru(updates.flatten(0, 1), slots.flatten(0, 1))
            slots = updated_slots.unflatten(0, slots.shape[:2])
        else:
            slots = slots + updates

        if self.mlp is not None:
            slots = self.mlp(slots)

        return slots, pre_norm_attn

    def forward(self, slots: torch.Tensor, features: torch.Tensor, n_iters = None):
        features = self.norm_features(features)
        keys = self.to_k(features)
        values = self.to_v(features)

        if n_iters is None:
            n_iters = self.n_iters

        for _ in range(n_iters):
            slots, pre_norm_attn = self.step(slots, keys, values)

        return {"slots": slots, "masks": pre_norm_attn}

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

        init_parameters(self, "xavier_uniform")
    
    def forward(self, q, k, v, attn_mask=None, return_seg=False):
        """
        q: B x N x D
        k: B x L x D
        v: B x L x D
        attn_mask: N x L
        return: B x N x L
        """
        B, N, _, = q.shape
        _, L, _ = k.shape
        seg_mask = None 

        # Project and split to different heads
        q = self.proj_q(q).view(B, N, self.num_heads, -1).transpose(1, 2) # [B, num_heads, N, D]
        k = self.proj_k(k).view(B, L, self.num_heads, -1).transpose(1, 2) # [B, num_heads, L, D]
        v = self.proj_v(v).view(B, L, self.num_heads, -1).transpose(1, 2) # [B, num_heads, L, D]     
        q = q * self.attn_scale 
        attn = torch.matmul(q, k.transpose(-1, -2)) # [B, num_heads, T, S]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1) 
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        # Inverted-attention + Renoemalization
        attn = F.softmax(attn, dim=-2) # normalize along query; -1 for key
        attn = attn / torch.sum(attn, dim=-1, keepdim=True)
        if return_seg:
            seg_mask = attn.detach().clone().mean(1).squeeze(1).permute(0, 2, 1)
        # Concat
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, -1)
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
        self.time_attn = MemEffAttention(
                dim=embed_dim,
                num_heads=time_nh
            )
            # nn.MultiheadAttention(
            #     embed_dim=embed_dim,
            #     num_heads=time_nh,
            #     batch_first=True,
            # )


        ## Object-Axis Self-Attention
        self.layer_obj = nn.LayerNorm(embed_dim)
        self.obj_attn = MemEffAttention(
                dim=embed_dim,
                num_heads=obj_nh
            )
            # nn.MultiheadAttention(
            #     embed_dim=embed_dim,
            #     num_heads=obj_nh,
            #     batch_first=True,
            # )

        ## Final Projection
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, in_dict: dict):
        features = in_dict["features"]
        x = features
        prev_slots = in_dict["prev_slots"]
        B, T, L, D = x.shape
        N = self.num_slots
        # B x T x N x D
        if prev_slots is None:
            prev_slots = self.init_slots.repeat(B, T, 1, 1)

        # Inverse Cross-Attention
        slots = self.ln_q(prev_slots)
        x = self.ln_kv(x)
        pos_emb = self.pos_emb(T, N).repeat(B, 1, 1, 1) # TODO : CHECK POS EMBED
        slots = pos_emb + slots

        slots = einops.rearrange(slots, 'b t n d -> b (t n) d')
        x = einops.rearrange(x, 'b t l d -> b (t l) d')
        slot_causal_mask = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1)
        slot_causal_mask = slot_causal_mask.unsqueeze(0).repeat(B, N, L).to(x.device)
        pred_slots, attn_mask = self.inverse_cross_attn(slots, x, x, attn_mask=slot_causal_mask, return_seg=self.get_mask) # B x T x N x D
        slots = slots + pred_slots # Residual Connection

        # Time-Axis Self-Attention        
        slots = einops.rearrange(slots, 'b (t n) d -> (b n) t d', b=B, n=N, t=T)
        slots = self.layer_time(slots) #  BxT, N, D
        # time_causal_mask = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1).to(x.device)
        pred_slots = self.time_attn(slots, attn_mask=True)#, slots, slots, attn_mask=time_causal_mask, is_causal=True)
        slots = slots + pred_slots # Residual Connection

        # Object-Axis Self-Attention
        slots = einops.rearrange(slots, '(b n) t d -> (b t) n d', b=B, n=N, t=T)
        slots = self.layer_obj(slots)
        pred_slots = self.obj_attn(slots)#, slots, slots)
        slots = slots + pred_slots # Residual Connection

        # Final Projection
        slots = slots.view(B, T, N, D)
        pred_slots = self.proj(slots)
        slots = slots +  pred_slots

        out_dict = {
            "features": features,
            "prev_slots": slots,
            "masks": None
        }
        if self.get_mask:
            out_dict["masks"] = attn_mask
        return out_dict

class CoordinatePositionEmbed(nn.Module):
    """Coordinate positional embedding as in Slot Attention."""

    def __init__(self, dim: int, size: Tuple[int, int], proj_dim: int = None):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.register_buffer("grid", self.build_grid(size))
        self.proj = nn.Conv2d(self.grid.shape[0], dim, kernel_size=1, bias=True)
        init_parameters(self.proj, "xavier_uniform")
        if proj_dim is not None:
            self.proj_dim = proj_dim
            self.proj_linear = nn.Linear(dim, proj_dim)

    @staticmethod
    def build_grid(
        size: Tuple[int, int],
        bounds: Tuple[float, float] = (-1.0, 1.0),
        add_inverse: bool = False,
    ) -> torch.Tensor:
        ranges = [torch.linspace(*bounds, steps=res) for res in size]
        grid = torch.meshgrid(*ranges, indexing="ij")

        if add_inverse:
            grid = torch.stack((grid[0], grid[1], 1.0 - grid[0], 1.0 - grid[1]), axis=0)
        else:
            grid = torch.stack((grid[0], grid[1]), axis=0)

        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.ndim == 4
        ), f"Expected input shape (batch, channels, height, width), but got {x.shape}"
        x = x + self.proj(self.grid)
        if hasattr(self, "proj_dim"):
            x = x.flatten(-2, -1)
            x = x.permute(0, 2, 1)
            x = self.proj_linear(x)
        return x