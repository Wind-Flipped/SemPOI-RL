import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import random
from utils import _L2_loss_mean
from GAT import GAT
from einops import repeat
from torchdiffeq import odeint
# import torchquad
from torch.distributions import Normal, Independent
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from trainer import top_np_recommendation

from LLMs import TravelStyleGenerator, TravelStyleRewardCalculator
import numpy as np

class PositionalEncoding(nn.Module):
    """正弦位置编码，用于给序列添加位置信息"""
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model] for batch_first=True
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MaskedAutoEncoder(nn.Module):
    """ Masked Autoencoder for Sequence Data
    """
    
    def __init__(self, seq_len=100, embed_dim=128, depth=6, num_heads=8,
                 decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding, +1 for cls token
        
        self.encoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.1,
                batch_first=True,
                norm_first=True
            ) for _ in range(depth)
        ])
        self.encoder_norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding, +1 for cls token
        
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(  # 使用Encoder作为Decoder
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=int(decoder_embed_dim * mlp_ratio),
                dropout=0.1,
                batch_first=True,
                norm_first=True
            ) for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True)  # decoder to original embedding
        # --------------------------------------------------------------------------
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # initialize position embeddings with sin-cos embedding
        pos_embed = self.get_1d_sincos_pos_embed(self.embed_dim, self.seq_len, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = self.get_1d_sincos_pos_embed(self.decoder_embed.out_features, self.seq_len, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # initialize cls token
        torch.nn.init.normal_(self.cls_token, std=.02)
        
        # initialize mask token
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def get_1d_sincos_pos_embed(self, embed_dim, seq_len, cls_token=False, temperature=10000.):
        """
        Create 1D sin-cos positional embeddings for sequences
        """
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(temperature) / embed_dim))
        
        pos_embed = np.zeros((seq_len, embed_dim))
        pos_embed[:, 0::2] = np.sin(position * div_term)
        pos_embed[:, 1::2] = np.cos(position * div_term)
        
        if cls_token:
            cls_pos_embed = np.zeros((1, embed_dim))
            pos_embed = np.concatenate([cls_pos_embed, pos_embed], axis=0)
        
        return pos_embed
    
    def random_masking(self, x, mask_ratio, hometown_len_list=None, destination_start_list=None, destination_end_list=None, valid_mask=None):
        """
        Perform per-sample random masking by per-sample shuffling for destination sequences.
        x: [N, L, D], sequence (hometown + destination concatenated)
        hometown_len_list: list of int, length of hometown sequence for each sample
        destination_start_list: list of int, start position of destination sequence in concatenated sequence
        destination_end_list: list of int, end position of destination sequence in concatenated sequence
        valid_mask: [N, L], True for valid positions, False for padding
        """
        N, L, D = x.shape  # batch, length, dim
        
        if valid_mask is None:
            valid_mask = torch.ones(N, L, dtype=torch.bool, device=x.device)
        
        # Create final masks for each sample
        final_masks = []
        ids_keep_list = []
        ids_restore_list = []
        x_masked_list = []
        
        for i in range(N):
            # Get destination sequence range for this sample
            if (hometown_len_list is not None and destination_start_list is not None and 
                destination_end_list is not None):
                dest_start = destination_start_list[i]
                dest_end = destination_end_list[i]
                
                # Create mask - only mask destination sequence (excluding start and end)
                mask = torch.zeros(L, dtype=torch.bool, device=x.device)
                
                if dest_end > dest_start + 1:  # At least one position between start and end
                    # Maskable positions: middle positions of destination sequence
                    maskable_positions = torch.arange(dest_start + 1, dest_end, device=x.device)
                    
                    # Filter by valid mask
                    maskable_positions = maskable_positions[valid_mask[i, maskable_positions]]
                    
                    if len(maskable_positions) > 0:
                        num_mask = int(len(maskable_positions) * mask_ratio)
                        if num_mask > 0:
                            # Random masking
                            noise = torch.rand(len(maskable_positions), device=x.device)
                            ids_shuffle = torch.argsort(noise)
                            
                            # Positions to mask
                            mask_indices = maskable_positions[ids_shuffle[:num_mask]]
                            mask[mask_indices] = True
                
                # Keep unmasked positions
                keep_indices = torch.nonzero(~mask, as_tuple=True)[0]
                ids_keep = keep_indices
            else:
                # Fallback to original logic if destination positions not provided
                valid_positions = torch.nonzero(valid_mask[i], as_tuple=True)[0]
                
                if len(valid_positions) == 0:
                    ids_keep = torch.arange(L, device=x.device)
                    mask = torch.zeros(L, dtype=torch.bool, device=x.device)
                else:
                    # Preserve first and last valid positions
                    if len(valid_positions) > 2:
                        start_pos = valid_positions[0]
                        end_pos = valid_positions[-1]
                        maskable_positions = valid_positions[1:-1]
                        
                        num_mask = int(len(maskable_positions) * mask_ratio)
                        mask = torch.zeros(L, dtype=torch.bool, device=x.device)
                        
                        if num_mask > 0:
                            noise = torch.rand(len(maskable_positions), device=x.device)
                            ids_shuffle = torch.argsort(noise)
                            mask_indices = maskable_positions[ids_shuffle[:num_mask]]
                            mask[mask_indices] = True
                    else:
                        mask = torch.zeros(L, dtype=torch.bool, device=x.device)
                    
                    keep_indices = torch.nonzero(~mask, as_tuple=True)[0]
                    ids_keep = keep_indices
            
            # Create restore indices (argsort of keep indices)
            ids_restore = torch.argsort(ids_keep)
            
            # Extract kept tokens
            x_masked = x[i, ids_keep, :]
            
            final_masks.append(mask)
            ids_keep_list.append(ids_keep)
            ids_restore_list.append(ids_restore)
            x_masked_list.append(x_masked)
        
        # Pad x_masked to same length for batch processing
        max_keep = max(x_m.shape[0] for x_m in x_masked_list)
        x_masked_padded = torch.zeros(N, max_keep, D, device=x.device)
        keep_mask = torch.zeros(N, max_keep, dtype=torch.bool, device=x.device)
        
        for i, x_m in enumerate(x_masked_list):
            length = x_m.shape[0]
            x_masked_padded[i, :length] = x_m
            keep_mask[i, :length] = True
        
        # Stack masks
        mask = torch.stack(final_masks, dim=0)  # [N, L]
        
        return x_masked_padded, mask, ids_keep_list, ids_restore_list, keep_mask
    
    def fixed_masking(self, x, hometown_len_list=None, destination_start_list=None, destination_end_list=None, valid_mask=None):
        """
        Fixed masking for inference: mask all destination positions except start and end
        x: [N, L, D], sequence (hometown + destination concatenated)
        hometown_len_list: list of int, length of hometown sequence for each sample
        destination_start_list: list of int, start position of destination sequence in concatenated sequence
        destination_end_list: list of int, end position of destination sequence in concatenated sequence
        valid_mask: [N, L], True for valid positions, False for padding
        """
        N, L, D = x.shape
        
        if valid_mask is None:
            valid_mask = torch.ones(N, L, dtype=torch.bool, device=x.device)
        
        final_masks = []
        ids_keep_list = []
        ids_restore_list = []
        x_masked_list = []
        
        for i in range(N):
            # Get destination sequence range for this sample
            if (hometown_len_list is not None and destination_start_list is not None and 
                destination_end_list is not None):
                dest_start = destination_start_list[i]
                dest_end = destination_end_list[i]
                
                # Create mask - mask all destination sequence except start and end
                mask = torch.zeros(L, dtype=torch.bool, device=x.device)
                
                if dest_end > dest_start + 1:  # At least one position between start and end
                    # Mask all middle positions of destination sequence
                    middle_positions = torch.arange(dest_start + 1, dest_end, device=x.device)
                    
                    # Filter by valid mask
                    middle_positions = middle_positions[valid_mask[i, middle_positions]]
                    mask[middle_positions] = True
                
                # Keep unmasked positions
                keep_indices = torch.nonzero(~mask, as_tuple=True)[0]
                ids_keep = keep_indices
            else:
                # Fallback to original logic if destination positions not provided
                valid_positions = torch.nonzero(valid_mask[i], as_tuple=True)[0]
                
                if len(valid_positions) == 0:
                    ids_keep = torch.arange(L, device=x.device)
                    mask = torch.zeros(L, dtype=torch.bool, device=x.device)
                else:
                    # Mask all middle valid positions, keep start and end
                    mask = torch.zeros(L, dtype=torch.bool, device=x.device)
                    if len(valid_positions) > 2:
                        start_pos = valid_positions[0]
                        end_pos = valid_positions[-1]
                        middle_positions = valid_positions[1:-1]
                        mask[middle_positions] = True
                    
                    keep_indices = torch.nonzero(~mask, as_tuple=True)[0]
                    ids_keep = keep_indices
            
            # Create restore indices
            ids_restore = torch.argsort(ids_keep)
            
            # Extract kept tokens
            x_masked = x[i, ids_keep, :]
            
            final_masks.append(mask)
            ids_keep_list.append(ids_keep)
            ids_restore_list.append(ids_restore)
            x_masked_list.append(x_masked)
        
        # Pad x_masked to same length for batch processing
        max_keep = max(x_m.shape[0] for x_m in x_masked_list)
        x_masked_padded = torch.zeros(N, max_keep, D, device=x.device)
        keep_mask = torch.zeros(N, max_keep, dtype=torch.bool, device=x.device)
        
        for i, x_m in enumerate(x_masked_list):
            length = x_m.shape[0]
            x_masked_padded[i, :length] = x_m
            keep_mask[i, :length] = True
        
        # Stack masks
        mask = torch.stack(final_masks, dim=0)  # [N, L]
        
        return x_masked_padded, mask, ids_keep_list, ids_restore_list, keep_mask
    
    def forward_encoder(self, x, mask_ratio=0.75, hometown_len_list=None, destination_start_list=None, 
                       destination_end_list=None, valid_mask=None, training=True):
        """
        Forward through encoder
        x: [N, L, D]
        hometown_len_list: list of int, length of hometown sequence for each sample
        destination_start_list: list of int, start position of destination sequence
        destination_end_list: list of int, end position of destination sequence
        valid_mask: [N, L], True for valid positions
        """
        # Add pos embed (without cls token)
        if x.shape[1] <= self.pos_embed.shape[1] - 1:  # -1 for cls token
            x = x + self.pos_embed[:, 1:x.shape[1] + 1, :]  # Skip cls token pos embed
        else:
            # Handle sequences longer than expected
            pos_embed_extended = self.pos_embed[:, 1:, :].repeat(1, (x.shape[1] // (self.pos_embed.shape[1] - 1)) + 1, 1)
            x = x + pos_embed_extended[:, :x.shape[1], :]
        
        # Masking
        if training:
            x_masked, mask, ids_keep_list, ids_restore_list, keep_mask = self.random_masking(
                x, mask_ratio, hometown_len_list, destination_start_list, destination_end_list, valid_mask
            )
        else:
            x_masked, mask, ids_keep_list, ids_restore_list, keep_mask = self.fixed_masking(
                x, hometown_len_list, destination_start_list, destination_end_list, valid_mask
            )
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # cls token + cls pos embed
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)
        
        # Update keep_mask to account for cls token
        cls_mask = torch.ones(x_masked.shape[0], 1, dtype=torch.bool, device=x_masked.device)
        keep_mask = torch.cat((cls_mask, keep_mask), dim=1)
        
        # Apply Transformer blocks
        for blk in self.encoder_blocks:
            x_masked = blk(x_masked, src_key_padding_mask=~keep_mask)
        
        x_masked = self.encoder_norm(x_masked)
        
        return x_masked, mask, ids_keep_list, ids_restore_list, keep_mask
    
    def forward_decoder(self, x_encoded, ids_keep_list, ids_restore_list, keep_mask, original_length):
        """
        Forward through decoder
        """
        N = len(ids_keep_list)
        
        # Embed tokens
        x = self.decoder_embed(x_encoded)
        
        # Append mask tokens to sequence (similar to MaskedAutoencoderViT)
        mask_tokens_list = []
        x_no_cls_list = []
        
        for i in range(N):
            ids_keep = ids_keep_list[i]
            
            # Remove cls token from encoded features
            x_no_cls = x[i, 1:keep_mask[i].sum(), :]  # Skip cls token, get valid encoded tokens
            x_no_cls_list.append(x_no_cls)
            
            # Calculate number of mask tokens needed
            num_mask_tokens = original_length - len(ids_keep)
            if num_mask_tokens > 0:
                mask_tokens = self.mask_token.repeat(1, num_mask_tokens, 1).squeeze(0)  # [num_mask, D]
            else:
                mask_tokens = torch.empty(0, self.mask_token.shape[-1], device=x.device)
            
            mask_tokens_list.append(mask_tokens)
        
        # Concatenate encoded tokens and mask tokens for each sample
        x_full_list = []
        for i in range(N):
            x_no_cls = x_no_cls_list[i]
            mask_tokens = mask_tokens_list[i]
            ids_keep = ids_keep_list[i]
            
            # Concatenate and unshuffle
            x_concat = torch.cat([x_no_cls, mask_tokens], dim=0)  # [L, D]
            
            # Create indices for unshuffling (restore original order)
            ids_restore = torch.argsort(torch.cat([ids_keep, torch.tensor([j for j in range(original_length) if j not in ids_keep], device=x.device)]))
            
            # Unshuffle to restore original sequence order
            x_full = x_concat[ids_restore]  # [L, D]
            
            x_full_list.append(x_full)
        
        # Stack to create batch and add cls token back
        x_full = torch.stack(x_full_list, dim=0)  # [N, L, D]
        
        # Add cls token back (at the beginning)
        cls_tokens = x[:, :1, :]  # [N, 1, D] - keep cls token from encoder
        x_full = torch.cat([cls_tokens, x_full], dim=1)  # [N, L+1, D]
        
        # Add pos embed
        if x_full.shape[1] <= self.decoder_pos_embed.shape[1]:
            x_full = x_full + self.decoder_pos_embed[:, :x_full.shape[1], :]
        else:
            # Handle sequences longer than expected
            pos_embed_extended = self.decoder_pos_embed.repeat(1, (x_full.shape[1] // self.decoder_pos_embed.shape[1]) + 1, 1)
            x_full = x_full + pos_embed_extended[:, :x_full.shape[1], :]
        
        # Apply Transformer blocks (using Encoder as Decoder)
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        
        x_full = self.decoder_norm(x_full)
        
        # Predictor projection
        x_full = self.decoder_pred(x_full)  # [N, L+1, embed_dim]
        
        # Remove cls token from output
        x_full = x_full[:, 1:, :]  # [N, L, embed_dim]
        
        return x_full
    
    def forward_loss(self, original, pred, mask, valid_mask=None):
        """
        Compute loss only on masked positions
        original: [N, L, D]
        pred: [N, L, D]
        mask: [N, L], True for masked positions
        valid_mask: [N, L], True for valid positions
        """
        if valid_mask is None:
            valid_mask = torch.ones_like(mask, dtype=torch.bool)
        
        # Only compute loss on masked AND valid positions
        loss_mask = mask & valid_mask  # [N, L]
        
        if loss_mask.sum() == 0:
            return torch.tensor(0.0, device=original.device)
        
        # Compute MSE loss
        loss = F.mse_loss(pred, original, reduction='none')  # [N, L, D]
        loss = loss.mean(dim=-1)  # [N, L], mean loss per position
        
        # Apply mask and compute mean loss on masked positions
        loss = (loss * loss_mask.float()).sum() / loss_mask.sum()
        
        return loss
    
    def forward(self, x, mask_ratio=0.75, hometown_len_list=None, destination_start_list=None, 
               destination_end_list=None, valid_mask=None, training=True):
        """
        Forward pass
        x: [N, L, D] input sequence (hometown + destination concatenated)
        hometown_len_list: list of int, length of hometown sequence for each sample
        destination_start_list: list of int, start position of destination sequence
        destination_end_list: list of int, end position of destination sequence
        valid_mask: [N, L] mask indicating valid positions (True for valid, False for padding)
        """
        original_length = x.shape[1]
        
        # Encoder
        latent, mask, ids_keep_list, ids_restore_list, keep_mask = self.forward_encoder(
            x, mask_ratio, hometown_len_list, destination_start_list, destination_end_list, valid_mask, training
        )
        
        # Decoder
        pred = self.forward_decoder(latent, ids_keep_list, ids_restore_list, keep_mask, original_length)
        
        # Loss (only during training)
        if training:
            loss = self.forward_loss(x, pred, mask, valid_mask)
            return loss, pred, mask
        else:
            return pred, mask


class Encoder(nn.Module):
    """Encoder mapping context sequences to parameters of the posterior q(z1)."""

    def __init__(
            self, poi_size,
            d_z: int,
            d_model: int, n_attn_heads: int, n_tf_layers: int, dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.time_proj = nn.Linear(1, d_model, bias=False)
        self.space_proj = nn.Linear(2, d_model, bias=False)
        self.poi_proj = nn.Linear(d_model, d_model, bias=False)
        self.poi_emb = POIEmbeddings(poi_size, d_model)
        self.transformer_stack = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_attn_heads,
                dim_feedforward=2 * d_model,
                batch_first=True,
                dropout=dropout_prob,
                norm_first=True,  # 使用Pre-LN架构更稳定
                activation=F.gelu,  # 明确指定激活函数
            ) for _ in range(n_tf_layers)
        ])

        self.gamma_proj = nn.Linear(d_model, d_z)
        self.tau_proj = nn.Linear(d_model, d_z)

        self.agg_token = nn.Parameter(torch.empty((1, 1, d_model)))
        nn.init.xavier_uniform_(self.agg_token)

    def forward(self, d_t, d_l, d_emb, d_pad):
        """Maps context sequences `x` to parameters of the posterior q(z1)."""

        t_emb = self.time_proj(d_t.to(torch.float32).unsqueeze(-1))
        coords_emb = self.space_proj(d_l.to(torch.float32))
        poi_emb = self.poi_emb(d_emb)

        x = torch.cat(
            [
                t_emb + coords_emb + poi_emb,
                repeat(self.agg_token, "() () d -> b () d", b=d_t.shape[0]), ],
            dim=1,
        )

        # PyTorch的src_key_padding_mask语义：True表示需要被忽略的位置
        # 因此需要将d_pad取反（假设d_pad中True表示有效位置）
        padding_mask = ~d_pad

        # 调试信息和数值稳定性保护
        for i, layer in enumerate(self.transformer_stack):
            try:
                x_before = x.clone()
                x_new = layer(x, src_key_padding_mask=padding_mask)

                # 检查输出是否有效
                if torch.isnan(x_new).any() or torch.isinf(x_new).any():
                    print(f"Warning: NaN/Inf detected in transformer layer {i}")
                    print(f"Input range: [{x_before.min():.4f}, {x_before.max():.4f}]")
                    print(f"d_pad shape: {d_pad.shape}, unique values: {torch.unique(d_pad)}")
                    # 保持原来的x不变，跳过这一层
                    continue
                else:
                    x = x_new

            except Exception as e:
                print(f"Error in transformer layer {i}: {e}")
                # 跳过这一层，使用原来的x
                continue

        x = x[:, -1, :]

        return x, self.gamma_proj(x), torch.nn.functional.softplus(self.tau_proj(x))  # 更稳定


def _nearest_interpolate(t_eval, t, z, ind_left, ind_right):
    dist_left = torch.abs(t_eval - t[ind_left])
    dist_right = torch.abs(t_eval - t[ind_right])
    nearer_right = dist_right < dist_left
    return torch.where(nearer_right.unsqueeze(1), z[ind_right], z[ind_left])


def _linear_interpolate(t_eval, t, z, ind_left, ind_right):
    t_left = t[ind_left]
    t_right = t[ind_right]
    weight_right = (t_eval - t_left) / (t_right - t_left + 1e-3)
    weight_left = 1 - weight_right
    return weight_left.unsqueeze(1) * z[ind_left] + weight_right.unsqueeze(1) * z[ind_right]


def interpolate(t_eval, t, z, method: str = "nearest"):
    """
    Interpolates values at specified evaluation points.

    Args:
        t_eval (Tensor): The evaluation time points, shape (n,).
        t (Tensor): The trajectory time points, shape (time,).
        z (Tensor): The trajectory values at time points `t`, shape (time, d_z).
        method (str, optional): The interpolation method ('nearest' or 'linear'). Defaults to 'nearest'.

    Returns:
        Tensor: Interpolated values at `t_eval`.
    """
    if method not in {"nearest", "linear"}:
        raise ValueError(f"Interpolation method {method} is not supported.")

    ind_right = torch.searchsorted(t, t_eval)  # 查找 t_eval 在时间序列 t 中的插入位置，返回的 ind_right 是右侧的索引
    ind_left = ind_right - 1
    ind_left.clamp_(min=0)
    ind_right.clamp_(max=len(t) - 1)

    if method == "nearest":
        return _nearest_interpolate(t_eval, t, z, ind_left, ind_right)
    else:  # method == "linear"
        return _linear_interpolate(t_eval, t, z, ind_left, ind_right)


def kl_norm_norm(mu0, mu1, sig0, sig1):
    """Calculates KL divergence between two K-dimensional Normal
        distributions with diagonal covariance matrices.

    Args:
        mu0: Mean of the first distribution. Has shape (*, K).
        mu1: Mean of the second distribution. Has shape (*, K).
        sig0: Diagonal of the covatiance matrix of the first distribution. Has shape (*, K).
        sig1: Diagonal of the covatiance matrix of the second distribution. Has shape (*, K).

    Returns:
        KL divergence between the distributions. Has shape (*, 1).
    """
    assert mu0.shape == mu1.shape == sig0.shape == sig1.shape, (
        f"{mu0.shape=} {mu1.shape=} {sig0.shape=} {sig1.shape=}")
    a = (sig0 / sig1).pow(2).sum(-1, keepdim=True)
    b = ((mu1 - mu0).pow(2) / sig1 ** 2).sum(-1, keepdim=True)
    c = 2 * (torch.log(sig1) - torch.log(sig0)).sum(-1, keepdim=True)
    kl = 0.5 * (a + b + c - mu0.shape[-1])
    return kl


def create_mlp(
        input_size,
        output_size,
        hidden_size,
        num_hidden_layers,
        activation_func,
        use_layer_norm=False,
        use_dropout=False,
        dropout_prob=0.5,
):
    """
    Create MLP with optional layer normalization and dropout.

    Args:
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.
        hidden_size (int): The size of the hidden layers.
        num_hidden_layers (int): The number of hidden layers.
        activation_func (function): The nonlinear activation function to use.
        use_layer_norm (bool): Whether to use layer normalization (default: False).
        use_dropout (bool): Whether to use dropout (default: False).
        dropout_prob (float): Dropout probability, used if use_dropout is True (default: 0.5).

    Returns:
        nn.Sequential: The constructed MLP model.
    """
    layers = []
    for i in range(num_hidden_layers):
        if i == 0:
            layers.append(nn.Linear(input_size, hidden_size))
        else:
            layers.append(nn.Linear(hidden_size, hidden_size))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(activation_func())
        if use_dropout:
            layers.append(nn.Dropout(dropout_prob))

    layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)


class DynamicsFunction(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, t, z):
        return self.f(z)


class DynamicTimeGenerator(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(DynamicTimeGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)
        self.fc_interval = nn.Linear(hidden_dim, 1)
        self.init_input = nn.Parameter(torch.zeros(1, input_dim), requires_grad=False)

    def forward(self, context, num_pred_list):
        batch_size = context.size(0)
        device = context.device

        full_times_list = []
        seq_lengths = []

        for i in range(batch_size):
            n_pred = num_pred_list[i]
            hidden = context[i:i + 1]
            rnn_input = self.init_input
            cum_time = torch.zeros(1, device=device)
            times = []
            for _ in range(n_pred):
                hidden = self.rnn_cell(rnn_input, hidden)
                delta = F.softplus(self.fc_interval(hidden))
                cum_time = cum_time + delta.squeeze(1)
                times.append(cum_time.clone())
            if len(times) > 0:
                times_tensor = torch.cat(times, dim=0)
            else:
                times_tensor = torch.tensor([], device=device)
            if times_tensor.numel() > 0:
                times_tensor = times_tensor / (times_tensor[-1] + 1e-6)
            full_time = torch.cat(
                [torch.tensor([0.0], device=device), times_tensor, torch.tensor([1.0], device=device)], dim=0)
            full_times_list.append(full_time)
            seq_lengths.append(full_time.numel())

        max_len = max(seq_lengths)
        full_times_padded = torch.zeros(batch_size, max_len, device=device)
        for i, t_seq in enumerate(full_times_list):
            length = t_seq.numel()
            full_times_padded[i, :length] = t_seq

        return full_times_padded


class ContinuousDecoder(nn.Module):
    """Maps latent state z(t) and spatial coordinate x to u(t, x).

    Attributes:
        d_z (int): Dimensionality of the latent state.
        d_x (int): Dimensionality of the spatial coodinates.
        d_u (int): Dimensionality of the latent spatiotemporal state.
        f (Module): Mapping from (z(t), x) to u(t, x).
    """

    def __init__(self, d_z, d_x, f, interp_method):
        super().__init__()
        # self.space_proj = nn.Linear(d_x, d_z, bias=False)
        self.f = f  # mlp
        self.interp_method = interp_method

    def forward(self, t_eval, t, z):
        """Evaluates the latent spatiotemporal state u(t, x) for a single trajectory t, z.

        Args:
            t_eval: Evaluation time points, has shape (n, ).
            t: Trajectory time points, has shape (time, ).
            z: Trajectory values at time points `t`, has shape (time, d_z).

        Returns:
            Latent spatiotemporals state at (t_eval, x_eval). Has shape (n, d_u).
        """
        if t_eval.ndim != 1 or t.ndim != 1:
            raise ValueError("t and t_eval should be a 1-dimensional arrays.")
        if z.ndim != 2:
            raise ValueError("z should be a 2-dimensional arrays.")
        if t.shape[0] != z.shape[0]:
            raise ValueError("t and z must have matching first dimension.")

        z_eval = interpolate(t_eval, t, z, method=self.interp_method)
        # return self.f(z_eval + self.space_proj(x_eval))
        return self.f(z_eval)


class IntensityCorrection(nn.Module):
    def __init__(self, val=0):
        super().__init__()
        self.val = val

    def forward(self, x):
        # return torch.pow(x, 2) + self.val
        return torch.exp(x) + self.val


class POIEmbeddings(nn.Module):
    def __init__(self, poi_size, poi_embed_dim):
        super(POIEmbeddings, self).__init__()
        self.emb = nn.Embedding(poi_size, poi_embed_dim)

    def forward(self, traj):
        x = self.emb(traj)
        return x


# =================Transformer framework================== #
class TransformerModel(nn.Module):
    def __init__(self, embed_size, nhead, nhid, nlayers, dropout=0.3):
        super(TransformerModel, self).__init__()

        # self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # src = src * math.sqrt(self.embed_size)
        # src = self.pos_encoder(src)
        x = self.transformer_encoder(src)

        return x


class Recommender(nn.Module):
    def __init__(self, out_dim, poi_size):
        super(Recommender, self).__init__()
        self.fc = nn.Linear(out_dim, poi_size)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, outputs):
        x = self.fc(outputs)
        x = self.leaky_relu(x)
        return x


# ============================Penalty=============================== #
class Drifting(nn.Module):
    def __init__(self, beta):
        super(Drifting, self).__init__()
        self.beta = beta

    def forward(self, fix_outputs, region_mask):

        batch_size, seq_len, _ = fix_outputs.size()
        max_num_moves = seq_len - 1
        total_similarity = 0.0

        count = 0

        for num_moves in range(1, max_num_moves + 1):
            for i in range(batch_size):
                valid_indices = region_mask[i]  # [poi_size]
                for t in range(seq_len - num_moves):
                    vec1 = fix_outputs[i, t, :][valid_indices]
                    vec2 = fix_outputs[i, t + num_moves, :][valid_indices]
                    sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
                    total_similarity += sim.item()
                    count += 1
        avg_similarity = torch.tensor(total_similarity / count)
        # for num_moves in range(1, max_num_moves + 1):
        #     shift_outputs = fix_outputs[:, num_moves:]
        #     similarity = F.cosine_similarity(fix_outputs[:, :-num_moves], shift_outputs, dim=-1)
        #     total_similarity += similarity.mean()
        # avg_similarity = total_similarity / (seq_len - 1)
        repetition_penalty_loss = -torch.log(1 - 0.5 * (avg_similarity + 1)) * self.beta

        return repetition_penalty_loss


class Guiding(nn.Module):
    def __init__(self, out_dim, poi_size):
        super(Guiding, self).__init__()
        self.predictor = Recommender(out_dim, poi_size)
        # self.confidence = nn.Linear(poi_size, 1)

    def forward(self, outputs, AM, PM):
        fix_outputs = self.predictor(outputs)  # [b,l,d] -> [b,l,v]
        clipped_PM = PM[:, :fix_outputs.shape[1]]  # [v,l_max] -> [v,l]
        clipped_outputs = fix_outputs * (clipped_PM.T.unsqueeze(0).expand(fix_outputs.shape[0], -1, -1))  # [b,l,v]

        return clipped_outputs


# Construct total framework(AR-Trip)
class SPOTModel(nn.Module):
    def __init__(self, args, poi_size, region_poi,
                 max_length_venue_id=100, max_length_ori_id=100, d_model=128, n_head=4, num_encoder_layers=1, n_tf_layers=4, d_z=128,
                 kg_dataset=None):

        super(SPOTModel, self).__init__()
        # initial LLMs
        self.travel_style_reward_calculator = TravelStyleRewardCalculator()
        self.travel_style_generator = TravelStyleGenerator(use_vllm=args.use_vllm, use_lora=args.use_lora, lora_path=args.lora_path)
        # initial hyperparameter
        self.hidden_size = d_model
        self.args = args
        self.llm_embedding_dim = args.llm_embedding_dim if args.use_llm else 0
        # model setting
        self.poi_embedding = POIEmbeddings(poi_size, self.hidden_size)
        self.poi_size = poi_size
        self.pos_emb = nn.Embedding(max_length_venue_id, self.hidden_size)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size * 4),
            nn.SiLU()
        )
        if self.args.kg:
            self.kg_dataset = kg_dataset
            self.n_entities = self.kg_dataset.entity_count
            self.n_relations = self.kg_dataset.relation_count
            self.entity_embedding = nn.Embedding(self.n_entities + 1, d_model)
            self.relations_embedding = nn.Embedding(self.n_relations + 1, d_model)
            self.kg_dict, self.poi2relations = self.kg_dataset.get_kg_dict(self.poi_size)
            self.gat = GAT(self.hidden_size, self.hidden_size, dropout=0.4, alpha=0.2).train()
            if self.args.trans == 'transr':
                self.projection_matrix = nn.Linear(self.hidden_size, self.args.projection_dim)

        self.encoder = Encoder(poi_size, d_z, d_model, n_head, n_tf_layers)
        self.dyf = DynamicsFunction(
            f=create_mlp(input_size=args.hidden_size,
                         output_size=args.hidden_size,
                         hidden_size=args.dyn_latent_dim,
                         num_hidden_layers=args.dyn_hid_layers,
                         activation_func=nn.GELU))
        self.time_generator = DynamicTimeGenerator(self.hidden_size, self.hidden_size)
        self.lm = nn.Sequential(
            create_mlp(
                input_size=args.hidden_size,
                output_size=1,
                hidden_size=args.lm_latent_dim,
                num_hidden_layers=args.lm_hid_layers,
                activation_func=nn.GELU,
            ),
            IntensityCorrection(0.0000001),
        )
        
        # 使用nn.Transformer进行序列到序列的预测
        self.seq2seq_transformer = nn.Transformer(
            d_model=self.hidden_size,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,
            dim_feedforward=4 * self.hidden_size,
            dropout=0.1,
            batch_first=True,
            norm_first=True  # 使用Pre-LN架构更稳定
        )
        self.seq_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.seq_norm = nn.LayerNorm(self.hidden_size)
        
        # 时间和空间的Embedding层用于st_module
        self.time_embedding = nn.Linear(1, self.hidden_size, bias=False)  # 时间是1维的
        self.space_embedding = nn.Linear(2, self.hidden_size, bias=False)  # 空间是2维的
        
        # 为st_module添加位置编码
        self.src_pos_encoding = PositionalEncoding(self.hidden_size, dropout=0.1, max_len=max_length_ori_id + max_length_venue_id + 10)
        self.tgt_pos_encoding = PositionalEncoding(self.hidden_size, dropout=0.1, max_len=max_length_ori_id + max_length_venue_id + 10)
        
        # 用于将拼接后的时间+空间+类别信息映射到统一维度
        self.concat_to_unified = nn.Linear(3 * self.hidden_size, self.hidden_size, bias=False)
        
        # 为Masked AutoEncoder添加可学习的mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        nn.init.xavier_uniform_(self.mask_token)
        
        # 初始化MaskedAutoEncoder
        if self.args.st_module:
            max_seq_len = max_length_ori_id + max_length_venue_id
            self.mae = MaskedAutoEncoder(
                seq_len=max_seq_len,
                embed_dim=self.hidden_size,
                depth=6,
                num_heads=n_head,
                decoder_embed_dim=self.hidden_size,
                decoder_depth=4,
                decoder_num_heads=n_head,
                mlp_ratio=4.0
            )

        self.transformer_encoder = TransformerModel(embed_size=self.hidden_size * 2, nhead=n_head,
                                                    nhid=self.hidden_size * 8, nlayers=num_encoder_layers)
        self.transformer_encoder2 = TransformerModel(embed_size=self.hidden_size, nhead=n_head,
                                                    nhid=4 * self.hidden_size, nlayers=num_encoder_layers)
        self.infer_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU()
        )
        if self.args.st_module and self.args.use_llm:
            self.predictor = Recommender(self.hidden_size * 2 + self.llm_embedding_dim, poi_size)
        elif self.args.ode and self.args.use_llm:
            self.predictor = Recommender(self.hidden_size * 2 + self.llm_embedding_dim, poi_size)
        elif self.args.ode and self.args.s_infer:
            self.predictor = Recommender(self.hidden_size * 4, poi_size)
        elif self.args.ode or self.args.s_infer:
            self.predictor = Recommender(self.hidden_size * 3, poi_size)
        else:
            self.predictor = Recommender(self.hidden_size * 2, poi_size)
        self.region_poi = region_poi
        self.region_embedding = nn.Embedding(len(self.region_poi), self.hidden_size)
        self.region_masks = {}
        for region, poi_list in region_poi.items():
            mask = torch.zeros(poi_size, dtype=torch.bool)
            mask[list(poi_list)] = True
            self.region_masks[region] = mask
        self.head_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.tail_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index during loss calculation

    def generate_square_subsequent_mask(self, sz):
        """生成causal mask，防止未来位置的信息泄露"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate_mask_for_mae(self, batch_size, seq_len, mask_ratio=0.75, preserve_start_end=True, training=True):
        """
        生成用于Masked AutoEncoder的掩码
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            mask_ratio: 掩码比例 (默认0.75)
            preserve_start_end: 是否保留起点和终点不被掩码 (默认True)
            training: 是否为训练模式 (训练时随机掩码，推理时固定掩码)
        Returns:
            mask: bool tensor [batch_size, seq_len], True表示被掩码的位置
        """
        device = next(self.parameters()).device
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            if preserve_start_end and seq_len > 2:
                # 保留起点(位置0)和终点(位置seq_len-1)不被掩码
                maskable_positions = list(range(1, seq_len - 1))
                if training:
                    # 训练时：随机掩码中间位置
                    num_mask = int(len(maskable_positions) * mask_ratio)
                    if num_mask > 0:
                        masked_indices = torch.randperm(len(maskable_positions))[:num_mask]
                        masked_positions = [maskable_positions[idx] for idx in masked_indices]
                        mask[i, masked_positions] = True
                else:
                    # 推理时：固定掩码策略，掩码除起点终点外的所有位置
                    mask[i, 1:-1] = True
            else:
                # 不保留起点终点的情况
                if training:
                    # 训练时：随机掩码
                    num_mask = int(seq_len * mask_ratio)
                    if num_mask > 0:
                        masked_indices = torch.randperm(seq_len)[:num_mask]
                        mask[i, masked_indices] = True
                else:
                    # 推理时：掩码所有位置
                    mask[i, :] = True
        
        return mask

    def calc_kg_loss_transE(self, h, r, pos_t, neg_t):
        """
        Calculates the loss for the model using the TransE approach.
        Args:
            h:      (kg_batch_size)
            r:      (kg_batch_size)
            pos_t:  (kg_batch_size)
            neg_t:  (kg_batch_size)
        Returns:
            loss
        """
        # Each sample corresponds to an index of a relation type, and embedding_relation converts the index of each relation type into the corresponding embedding vector.
        r_embed = self.relations_embedding(r)
        h_embed = self.poi_embedding(h)  # (kg_batch_size, entity_dim)
        pos_t_embed = self.entity_embedding(pos_t)  # (kg_batch_size, entity_dim)
        neg_t_embed = self.entity_embedding(neg_t)  # (kg_batch_size, entity_dim)
        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2),
                              dim=1)  # (kg_batch_size) As per the formula f_d in the paper.
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)  # (kg_batch_size)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        # This value can be considered as the "energy" of the input samples.
        # This code is typically used for calculating regularization terms in the loss function.
        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(
            neg_t_embed)
        # # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        return loss

    def calc_kg_loss_transR(self, h, r, pos_t, neg_t):
        """
        Calculates the loss for the model using the TransR approach.
        Args:
            h:      (kg_batch_size)
            r:      (kg_batch_size)
            pos_t:  (kg_batch_size)
            neg_t:  (kg_batch_size)
        Returns:
            loss
        """
        r_embed = self.projection_matrix(self.relations_embedding(r))
        h_embed = self.projection_matrix(self.poi_embedding(h))
        pos_t_embed = self.projection_matrix(self.entity_embedding(pos_t))
        neg_t_embed = self.projection_matrix(self.entity_embedding(neg_t))
        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(
            neg_t_embed)
        # # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        return loss

    def calc_kg_loss_SEEK(self, h, r, pos_t, neg_t):
        """
        Calculates the loss using the SEEK approach for knowledge graph embeddings.
        Args:
            h:      (kg_batch_size)
            r:      (kg_batch_size)
            pos_t:  (kg_batch_size)
            neg_t:  (kg_batch_size)
        Returns:
            loss
        """
        # Each sample corresponds to an index of a relation type, and the embedding_relation converts the index of each relation type into the corresponding embedding vector.
        r_embed = self.relations_embedding(r)  # (kg_batch_size, relation_dim)
        h_embed = self.poi_embedding(h)  # (kg_batch_size, entity_dim)
        pos_t_embed = self.entity_embedding(pos_t)  # (kg_batch_size, entity_dim)
        neg_t_embed = self.entity_embedding(neg_t)  # (kg_batch_size, entity_dim)

        k_num = self.args.segments
        rank = int(self.hidden_size / k_num)
        h = [h_embed[i * rank: (i + 1) * rank] for i in range(k_num)]
        h = tuple(h)
        r = [r_embed[i * rank: (i + 1) * rank] for i in range(k_num)]
        r = tuple(r)
        pos_t = [pos_t_embed[i * rank: (i + 1) * rank] for i in range(k_num)]
        pos_t = tuple(pos_t)
        neg_t = [neg_t_embed[i * rank: (i + 1) * rank] for i in range(k_num)]
        neg_t = tuple(neg_t)
        pos_tmp = 0
        neg_tmp = 0

        for x in range(k_num):
            for y in range(k_num):
                s = -1 if x % 2 != 0 and x + y >= k_num else 1
                w = y if x % 2 == 0 else (x + y) % k_num
                pos_tmp += s * r[x] * h[y] * pos_t[w]
                neg_tmp += s * r[x] * h[y] * neg_t[w]
        pos_score = torch.sum(pos_tmp, 1)
        neg_score = torch.sum(neg_tmp, 1)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        # This value can be considered as the "energy" of the input samples.
        # This code is typically used for calculating regularization terms in the loss function.
        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(
            neg_t_embed)
        # # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        # loss = kg_loss
        return loss

    def _alias(self, bids, n_poi):
        """
        Creates an alias tensor mapping original POI indices to a new set of indices.
        Returns:
            torch.Tensor
        """
        alias = torch.zeros(n_poi).long()
        for idx, b in enumerate(bids):
            alias[b] = idx
        alias = alias.to(self.args.device)
        return alias

    def _avg_pooling(self, ck, emb):
        """
        Applies average pooling to embeddings.
        Returns:
            Tensor
        """
        emb_sum = torch.sum(emb, axis=1)
        row_count = torch.sum(ck != 0, axis=-1)
        emb_agg = emb_sum / row_count.unsqueeze(1).expand_as(emb_sum)
        return emb_agg

    def _relation(self, emb, r, rel_embs, mode='transr'):
        """
        Applies a relation transformation to the embeddings, Dynamic Mapping.
        Args:
            o_emb: b x h
            d_emb: b x l x h
        Returns:
            Tensor
        """
        if mode == 'transr':
            relation = rel_embs[r].view(-1, self.hidden_size, self.hidden_size)  # b x (h x h)
            if len(emb.shape) == 2:
                emb_r = torch.bmm(emb.unsqueeze(1), relation).squeeze(1)
                return emb_r
            elif len(emb.shape) == 3:
                emb_r = torch.matmul(emb.unsqueeze(2), relation.unsqueeze(1).expand(-1, emb.size(1), -1, -1)).squeeze(2)
                return emb_r
        if mode == 'transd':
            relation = rel_embs[r]  # b x h Embeddings of 64 regions (cities) visited by users
            if len(emb.shape) == 2:
                # b x h x h matrix multiplication
                # equivalent to the embedding weights of the region (city) multiplied by the node embedding that has passed through a linear layer.
                trans_mat = torch.matmul(relation.unsqueeze(2), self.head_linear(emb).unsqueeze(1))  # b x h x h
                # torch.bmm() might be faster, but both are matrix-level multiplication
                emb_r = torch.bmm(emb.unsqueeze(1), trans_mat).squeeze(1)
                return emb_r
            elif len(emb.shape) == 3:
                # b x h x h (64, 13, 128, 1) * (64, 13, 1, 128)
                trans_mat = torch.matmul(relation.view(relation.size(0), 1, -1, 1).expand(-1, emb.size(1), -1, -1),
                                         self.tail_linear(emb).unsqueeze(2))  # b x h x h
                emb_r = torch.matmul(emb.unsqueeze(2), trans_mat).squeeze(2)
                return emb_r
        if mode == 'transe':
            return emb

    def drop_edge_random(self, poi2entities, p_drop, padding):
        """
        Randomly drops edges from the POI to entity mappings.
        Returns:
            dict
        """
        res = dict()
        for item, es in poi2entities.items():
            new_es = list()
            for e in es.tolist():
                if (random.random() > p_drop):
                    new_es.append(e)
                else:
                    new_es.append(padding)
            res[item] = torch.IntTensor(new_es).to(self.args.device)
        return res

    def get_kg_views(self):
        """
        Generates two views of the knowledge graph by randomly dropping edges.
        Returns:
            tuple
        """
        kg = self.kg_dict
        view1 = self.drop_edge_random(kg, self.args.kg_p_drop, self.n_entities)
        view2 = self.drop_edge_random(kg, self.args.kg_p_drop, self.n_entities)
        return view1, view2

    def cal_poi_embedding_mean(self, kg: dict):
        """
        Calculates the mean embeddings of POIs based on their associated entities.
        Returns:
            Tensor
        """
        poi_embs = self.poi_embedding(torch.IntTensor(list(kg.keys())).to(self.args.device))  # poi_num, emb_dim
        poi_entities = torch.stack(list(kg.values()))  # poi_num, entity_num_each
        entity_embs = self.entity_embedding(poi_entities)  # poi_num, entity_num_each, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(poi_entities != self.n_entities, torch.ones_like(poi_entities),
                                   torch.zeros_like(poi_entities)).float()
        # padding is zero
        entity_embs = entity_embs * padding_mask.unsqueeze(-1).expand(entity_embs.size())
        # poi_num, emb_dim
        entity_embs_sum = entity_embs.sum(1)
        entity_embs_mean = entity_embs_sum / padding_mask.sum(-1).unsqueeze(-1).expand(entity_embs_sum.size())
        # replace nan with zeros
        entity_embs_mean = torch.nan_to_num(entity_embs_mean)
        # poi_num, emb_dim
        return poi_embs + entity_embs_mean

    def cal_poi_embedding_gat(self, kg: dict):
        """
        Calculates the POI embeddings using a Graph Attention Network (GAT) based on the associated entities.
        Returns:
            Tensor
        """
        poi_embs = self.poi_embedding(torch.IntTensor(list(kg.keys())).to(self.args.device))  # poi_num, emb_dim
        poi_entities = torch.stack(list(kg.values()))  # poi_num, entity_num_each
        entity_embs = self.entity_embedding(poi_entities)  # poi_num, entity_num_each, emb_dim
        # poi_num, entity_num_each
        padding_mask = torch.where(poi_entities != self.n_entities, torch.ones_like(poi_entities),
                                   torch.zeros_like(poi_entities)).float()
        return self.gat(poi_embs, entity_embs, padding_mask)

    def cal_poi_embedding_rgat(self, kg: dict):
        """
        Calculates POI embeddings using a Relational Graph Attention Network (RGAT).
        Returns:
            Tensor
        """
        poi_embs = self.poi_embedding(torch.IntTensor(list(kg.keys())).to(self.args.device))  # poi_num, emb_dim
        poi_entities = torch.stack(list(kg.values()))  # poi_num, entity_num_each
        poi_relations = torch.stack(list(self.poi2relations.values()))
        entity_embs = self.entity_embedding(poi_entities)  # poi_num, entity_num_each, emb_dim
        relation_embs = self.relations_embedding(poi_relations)  # poi_num, entity_num_each, emb_dim
        padding_mask = torch.where(poi_entities != self.n_entities, torch.ones_like(poi_entities),
                                   torch.zeros_like(poi_entities)).float()
        return self.gat.forward_relation(poi_embs, entity_embs, relation_embs, padding_mask)

    def cal_poi_embedding_from_kg(self, kg: dict):
        """
        Calculates POI embeddings based on the specified knowledge graph convolution method.
        Returns:
            Tensor
        """
        if kg is None:
            kg = self.kg_dict

        if (self.args.kgcn == "GAT"):
            return self.cal_poi_embedding_gat(kg)
        elif self.args.kgcn == "RGAT":
            return self.cal_poi_embedding_rgat(kg)
        elif (self.args.kgcn == "MEAN"):
            return self.cal_poi_embedding_mean(kg)
        elif (self.args.kgcn == "NO"):
            return self.poi_embedding.weight

    def get_ui_views_weighted(self, poi_stabilities, stab_weight):
        """
        Calculates weighted POI views based on stability scores.
        Returns:
            Tensor
        """
        # kg probability of keep
        poi_stabilities = torch.exp(poi_stabilities)
        kg_weights = (poi_stabilities - poi_stabilities.min()) / (poi_stabilities.max() - poi_stabilities.min())
        # Replace elements in kg_weights less than or equal to 0.3 with 0.3, keep elements greater than 0.3 unchanged.
        kg_weights = kg_weights.where(kg_weights > 0.3, torch.ones_like(kg_weights) * 0.3)
        weights = (1 - self.args.ui_p_drop) / torch.mean(stab_weight * kg_weights) * (stab_weight * kg_weights)
        # weights = weights.where(weights>0.3, torch.ones_like(weights) * 0.3)
        # Replace elements in weights greater than or equal to 0.95 with 0.95, keep elements less than 0.95 unchanged.
        weights = weights.where(weights < 0.95, torch.ones_like(weights) * 0.95)
        # Perform Bernoulli sampling to get a mask tensor poi_mask of the same dimension as weights,
        # where the probability of an element being True is the corresponding value in weights.
        # Values are chosen as 1 or 0 with probabilities p and 1-p, respectively.
        poi_mask = torch.bernoulli(weights).to(torch.bool)
        # drop
        poi_mask.requires_grad = False
        return poi_mask

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        Calculates the similarity between two tensors.
        Returns:
            Tensor
        """
        if z1.size()[0] == z2.size()[0]:
            return F.cosine_similarity(z1, z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

    def poi_kg_stability(self, view1, view2):
        """
        Computes the stability of POI embeddings across two views of the knowledge graph.
        Returns:
            Tuple
        """
        kgv1_ro = self.cal_poi_embedding_from_kg(view1)
        kgv2_ro = self.cal_poi_embedding_from_kg(view2)
        sim = self.sim(kgv1_ro, kgv2_ro)
        return kgv1_ro, kgv2_ro, sim

    def get_views(self, aug_side="both"):
        """
        Generates augmented views for contrastive learning.
        Returns:
            Dict
        """
        # drop (epoch based)
        # kg drop -> 2 views -> view similarity for item
        # Randomly remove tail entities and fill in the removed parts.
        kgv1, kgv2 = self.get_kg_views()
        # [item_num]
        kgv1, kgv2, stability = self.poi_kg_stability(kgv1, kgv2)  # Calculate consistency
        kgv1 = kgv1.to(self.args.device)
        kgv2 = kgv2.to(self.args.device)
        stability = stability.to(self.args.device)
        # item drop -> 2 views
        # Delete the user-item interaction edges (deleting edges with item nodes as index) from the interaction graph.
        v1_mask = self.get_ui_views_weighted(stability, 1)
        # uiv2 = self.ui_drop_random(world.ui_p_drop)
        v2_mask = self.get_ui_views_weighted(stability, 1)

        contrast_views = {
            "kgv1": kgv1,
            "kgv2": kgv2,
            "uiv1": v1_mask,
            "uiv2": v2_mask
        }
        return contrast_views

    def info_nce_loss_overall(self, z1, z2):
        """
        Calculates the InfoNCE loss, a contrastive loss used for learning efficient embeddings.
        Returns:
            torch.Tensor
        """
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        batch_size, d_model = z1.shape
        features = torch.cat([z1, z2], dim=0)  # (batch_size * 2, d_model)

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [batch_size * 2, 1]

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [batch_size * 2, 2N-2]

        logits = torch.cat([positives, negatives], dim=1)  # (batch_size * 2, batch_size * 2 - 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)  # (batch_size * 2, 1)
        logits = logits / self.tau

        loss = criterion(logits, labels)
        return loss

    def pad_with_embedding(self, seq_list, pad_vector):
        # seq_list: list of tensors [seq_len, d]
        # pad_vector: tensor of shape [d]
        max_len = max(seq.shape[0] for seq in seq_list)
        padded_list = []
        for seq in seq_list:
            pad_len = max_len - seq.shape[0]
            if pad_len > 0:
                pad_tensor = pad_vector.unsqueeze(0).expand(pad_len, -1)
                padded_seq = torch.cat([seq, pad_tensor], dim=0)
            else:
                padded_seq = seq
            padded_list.append(padded_seq)
        return torch.stack(padded_list, dim=0)

    def forward(self, messages, o_ck, query, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg, d_rg, target_seq=None):
        batch_size, seq_length = query.size()
        # region_repr = self.region_embedding.weight
        region_mask = torch.stack([self.region_masks[int(r)] for r in d_rg], dim=0)
        region_mask_uns = region_mask.unsqueeze(1).expand(-1, query.size(1), -1).to(query.device)
        # initialize
        if self.args.kg:
            poi_embedding = self.cal_poi_embedding_from_kg(self.kg_dict)
            query_emb = poi_embedding[query]  # [b,l,d]
            o_emb = poi_embedding[o_ck]
            d_target_emb = poi_embedding[d_ck]
            pad_vec = poi_embedding[0]
        else:
            query_emb = self.poi_embedding(query)  # [b,l,d]
            o_emb = self.poi_embedding(o_ck)
            d_target_emb = self.poi_embedding(d_ck)
            pad_vec = self.poi_embedding.emb.weight[0]

        if self.args.use_llm:
            self.args.kg = False
            self.args.s_infer = False
            # 1. 通过LLM生成文本
            if self.args.use_target_llm:
                generated_texts = messages
            else:
                generated_texts = self.travel_style_generator.get_output(messages, max_length=512, temperature=0.7)
            # 2. 获取生成文本的embedding（截断到self.hidden_size维）
            generated_embeddings = self.travel_style_reward_calculator.get_embedding(generated_texts,
                                                                                     embedding_dim=self.llm_embedding_dim)
            generated_embeddings = torch.tensor(generated_embeddings).to(self.args.device)  # [b, d]
            # 对LLM embedding进行L2归一化
            generated_embeddings = F.normalize(generated_embeddings, p=2, dim=-1)
            P_L = generated_embeddings.unsqueeze(1).expand([generated_embeddings.shape[0], d_target_emb.shape[1], generated_embeddings.shape[1]])
        

        if self.args.st_module:
            # 初始化损失变量
            seq2seq_loss = torch.tensor(0.0, device=self.args.device, dtype=torch.float32)
            # 使用MaskedAutoEncoder进行序列到序列的预测
            self.args.ode = False
            
            # 修复维度不匹配问题：去除o_pad和d_pad的最后一个位置
            o_pad_fixed = o_pad[:, :-1]  # [b, o_seq_len] 去除最后一个位置
            d_pad_fixed = d_pad[:, :-1]  # [b, d_seq_len] 去除最后一个位置
            
            # 构建源序列（家乡序列）的完整表征
            o_time_emb = self.time_embedding(o_t.to(torch.float32).unsqueeze(-1))  # [b, o_seq_len, d]
            o_space_emb = self.space_embedding(o_l.to(torch.float32))  # [b, o_seq_len, d]
            o_concat_emb = torch.cat([o_time_emb, o_space_emb, o_emb], dim=-1)  # [b, o_seq_len, 3*d]
            o_seq = self.concat_to_unified(o_concat_emb)  # [b, o_seq_len, d] 映射到统一维度
            
            # 构建目标序列（目的地序列）的完整表征
            d_time_emb = self.time_embedding(d_t.to(torch.float32).unsqueeze(-1))  # [b, d_seq_len, d]
            d_space_emb = self.space_embedding(d_l.to(torch.float32))  # [b, d_seq_len, d]
            d_concat_emb = torch.cat([d_time_emb, d_space_emb, d_target_emb], dim=-1)  # [b, d_seq_len, 3*d]
            d_seq = self.concat_to_unified(d_concat_emb)  # [b, d_seq_len, d] 映射到统一维度
            
            # ======================== 完全拼接家乡和目的地序列 ========================
            # 将家乡序列和目标序列完全拼接，中间不留pad
            combined_seq_list = []
            combined_pad_list = []
            
            for i in range(batch_size):
                # 获取家乡序列的有效长度
                o_valid_len = o_pad_fixed[i].sum().item()
                o_seq_valid = o_seq[i, :o_valid_len, :]  # [o_valid_len, d]
                
                # 获取目的地序列的有效长度
                d_valid_len = d_pad_fixed[i].sum().item()
                d_seq_valid = d_seq[i, :d_valid_len, :]  # [d_valid_len, d]
                
                # 完全拼接（中间不留pad）
                combined_seq = torch.cat([o_seq_valid, d_seq_valid], dim=0)  # [o_valid_len + d_valid_len, d]
                combined_len = combined_seq.shape[0]
                
                # 创建对应的padding mask（True表示有效位置）
                combined_pad = torch.ones(combined_len, dtype=torch.bool, device=self.args.device)
                
                combined_seq_list.append(combined_seq)
                combined_pad_list.append(combined_pad)
            
            # 将序列pad到相同长度
            max_combined_len = max(seq.shape[0] for seq in combined_seq_list)
            combined_seq_padded = torch.zeros(batch_size, max_combined_len, self.hidden_size, device=self.args.device)
            combined_pad_padded = torch.zeros(batch_size, max_combined_len, dtype=torch.bool, device=self.args.device)
            
            # 记录每个样本的家乡序列长度，用于后续提取目的地部分
            o_valid_lens = []
            d_valid_lens = []
            
            for i, (seq, pad) in enumerate(zip(combined_seq_list, combined_pad_list)):
                seq_len = seq.shape[0]
                combined_seq_padded[i, :seq_len, :] = seq
                combined_pad_padded[i, :seq_len] = pad
                
                # 记录原始长度信息
                o_valid_len = o_pad_fixed[i].sum().item()
                d_valid_len = d_pad_fixed[i].sum().item()
                o_valid_lens.append(o_valid_len)
                d_valid_lens.append(d_valid_len)
            
            # ======================== 使用MaskedAutoEncoder ========================
            is_training = target_seq is not None
            
            # 计算目的地序列的起始和终止位置
            hometown_len_list = o_valid_lens  # 家乡序列长度
            destination_start_list = o_valid_lens  # 目的地起始位置 = 家乡序列长度
            destination_end_list = [o_valid_lens[i] + d_valid_lens[i] - 1 for i in range(batch_size)]  # 目的地终止位置
            
            if is_training:
                # 训练时：随机掩码目的地序列的部分，保留起点终点
                loss, pred, mask = self.mae(
                    combined_seq_padded,
                    mask_ratio=0.5,
                    hometown_len_list=hometown_len_list,
                    destination_start_list=destination_start_list,
                    destination_end_list=destination_end_list,
                    valid_mask=combined_pad_padded,
                    training=True
                )
                seq2seq_loss = loss
                
                # 提取目的地部分的预测结果用于后续处理
                P_D_list = []
                for i in range(batch_size):
                    o_len = o_valid_lens[i]
                    d_len = d_valid_lens[i]
                    if d_len > 0:
                        d_pred = pred[i, o_len:o_len+d_len, :]  # 提取目的地部分
                        # Pad到原始目的地序列长度
                        d_pred_padded = torch.zeros(d_pad_fixed.shape[1], self.hidden_size, device=self.args.device)
                        d_pred_padded[:d_len, :] = d_pred
                        P_D_list.append(d_pred_padded)
                    else:
                        # 如果没有有效的目的地序列，创建零填充
                        d_pred_padded = torch.zeros(d_pad_fixed.shape[1], self.hidden_size, device=self.args.device)
                        P_D_list.append(d_pred_padded)
                
                P_D = torch.stack(P_D_list, dim=0)  # [b, d_seq_len, d]
                
            else:
                # 推理时：mask除了目的地起点终点外的其他目的地序列部分
                pred, mask = self.mae(
                    combined_seq_padded,
                    mask_ratio=0.5,  # 这个参数在推理时不使用
                    hometown_len_list=hometown_len_list,
                    destination_start_list=destination_start_list,
                    destination_end_list=destination_end_list,
                    valid_mask=combined_pad_padded,
                    training=False
                )
                
                # 提取目的地部分的预测结果
                P_D_list = []
                for i in range(batch_size):
                    o_len = o_valid_lens[i]
                    d_len = d_valid_lens[i]
                    if d_len > 0:
                        d_pred = pred[i, o_len:o_len+d_len, :]  # 提取目的地部分
                        # Pad到原始目的地序列长度
                        d_pred_padded = torch.zeros(d_pad_fixed.shape[1], self.hidden_size, device=self.args.device)
                        d_pred_padded[:d_len, :] = d_pred
                        P_D_list.append(d_pred_padded)
                    else:
                        # 如果没有有效的目的地序列，创建零填充
                        d_pred_padded = torch.zeros(d_pad_fixed.shape[1], self.hidden_size, device=self.args.device)
                        P_D_list.append(d_pred_padded)
                
                P_D = torch.stack(P_D_list, dim=0)  # [b, d_seq_len, d]
            
            # 应用目的地序列的padding mask来屏蔽无效位置的输出
            P_D = P_D * d_pad_fixed.unsqueeze(-1)  # 将无效位置置零

        if self.args.ode:
            u_o_emb_d, gamma, tau = self.encoder(o_t, o_l, o_ck, o_pad)
            z_0 = gamma + tau * torch.randn_like(tau)
            dynamic_d_emb = self.encoder.time_proj(d_t.to(torch.float32).unsqueeze(-1)) + self.encoder.space_proj(
                d_l) + self.encoder.poi_emb(d_ck)
            # dynamic_d_emb = self.encoder.time_proj(d_t.to(torch.float32).unsqueeze(-1)) + self.encoder.poi_emb(d_ck)
            P_D = []
            process_loglik = torch.tensor([0.0], device=self.args.device, dtype=torch.float32)
            obs_loglik = torch.tensor([0.0], device=self.args.device, dtype=torch.float32)
            for j in range(batch_size):
                valid_idx = torch.nonzero(d_pad[j], as_tuple=True)[0][:-1]
                n_pred = len(valid_idx) - 2
                if target_seq is not None:
                    gt_times_unif = d_t[j][valid_idx].to(torch.float32)
                    # print("gt_times_unif:", gt_times_unif)
                    z_unif_j = odeint(self.dyf, z_0[j].unsqueeze(0), gt_times_unif,
                                      rtol=self.args.rtol, atol=self.args.atol, method=self.args.solver,
                                      options={"min_step": 0.0001, "max_step": 100})
                else:
                    s_unif = torch.linspace(0, 1, n_pred + 2, device=self.args.device, dtype=torch.float32)
                    z_unif_j = odeint(self.dyf, z_0[j].unsqueeze(0), s_unif,
                                      rtol=self.args.rtol, atol=self.args.atol, method=self.args.solver,
                                      options={"min_step": 0.0001, "max_step": 100})
                u_hat = z_unif_j.transpose(0, 1).squeeze(0)
                P_D.append(u_hat)
                if target_seq is not None:
                    lm_hat = self.lm(u_hat)
                    process_loglik += torch.sum(torch.log(lm_hat))
                    f_values = self.lm(z_unif_j).squeeze(-1).squeeze(-1)

                    integrated_value = torch.trapz(f_values, gt_times_unif)

                    process_loglik -= integrated_value

                    v = dynamic_d_emb[j][valid_idx]
                    obs_loglik += Normal(u_hat, self.args.sig_v).log_prob(v).sum()
            if target_seq is not None:
                kl_qp = 2 * kl_norm_norm(gamma, torch.zeros_like(gamma), tau, torch.ones_like(tau)).sum()
                elbo_loss = - (obs_loglik + process_loglik - kl_qp)
            P_D = self.pad_with_embedding(P_D, pad_vec)

        if self.args.s_infer:
            u_o_emb_s = self._avg_pooling(o_ck, o_emb)
            infer = self.infer_layer(u_o_emb_s)
            if target_seq is not None:
                u_d_emb_s = self._avg_pooling(d_ck, d_target_emb)
                infer_loss = torch.norm(infer - u_d_emb_s, p=2, dim=-1).mean()
            P_S = infer.unsqueeze(1).expand_as(d_target_emb)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=query.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embedded = self.pos_emb(position_ids)
        if self.args.use_llm:
            encoder_output = self.transformer_encoder2(position_embedded)
        else:
            model_input = torch.cat([query_emb, position_embedded], dim=2)
            encoder_output = self.transformer_encoder(model_input)
        # model_input = torch.cat([query_emb, position_embedded], dim=2)
        # encoder_output = self.transformer_encoder(model_input)

        if self.args.st_module and self.args.use_llm:
            encoder_output = torch.cat([encoder_output, P_D, P_L], dim=2)
        elif self.args.st_module:
            encoder_output = torch.cat([encoder_output, P_D], dim=2)
        elif self.args.ode and self.args.use_llm:
            encoder_output = torch.cat([encoder_output, P_D, P_L], dim=2)
        elif self.args.ode and self.args.s_infer:
            encoder_output = torch.cat([encoder_output, P_D, P_S], dim=2)
        elif self.args.ode:
            encoder_output = torch.cat([encoder_output, P_D], dim=2)
        elif self.args.s_infer:
            encoder_output = torch.cat([encoder_output, P_S], dim=2)
        poi_output = self.predictor(encoder_output)
        masked_poi_output = poi_output.masked_fill(~region_mask_uns, -1e9)
        if target_seq is not None:
            loss = self.criterion(masked_poi_output.view(-1, self.poi_size), d_ck.flatten())
            if self.args.ode:
                loss += elbo_loss.sum()
            if self.args.st_module:
                loss += seq2seq_loss
            if self.args.s_infer:
                loss += infer_loss
            return loss
        else:
            # _, predicted_ids = torch.max(masked_poi_output, dim=-1)
            guidance_similarity_ratio, guidance_candidate_ids = torch.topk(masked_poi_output,
                                                                           k=masked_poi_output.shape[1],
                                                                           dim=2)
            predicted_ids = top_np_recommendation(guidance_candidate_ids, guidance_similarity_ratio,
                                                  confidence=torch.tensor(self.args.confidence),
                                                  threshold=0.8)
            return predicted_ids