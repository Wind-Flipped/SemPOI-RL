import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import random

class MaskedAutoEncoder(nn.Module):
    """ Masked Autoencoder for Sequence Data with Semantic-aware Masking
    """
    
    def __init__(self, seq_len=100, embed_dim=128, depth=6, num_heads=8,
                 decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, num_semantic_parts=8):
        super().__init__()
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_semantic_parts = num_semantic_parts  # 语义部分数量
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.external_cls_token = None  # 用于存储外部LLM embedding作为cls_token
        self.semantic_parts_embedding = None  # 用于存储语义部分信息
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
    
    def set_external_cls_token(self, P_L):
        """
        设置外部 LLM embedding 作为 cls_token 并生成语义部分信息
        Args:
            P_L: [N, D] LLM embedding，如果 D > embed_dim 则截断，如果 D < embed_dim 则填充
        
        Note: 当 num_semantic_parts = 0 时，只设置 cls_token 但不生成语义部分信息，
              此时将使用普通的随机masking策略
        """
        if P_L is not None:
            # 处理维度不匹配问题
            if P_L.shape[-1] > self.embed_dim:
                P_L_processed = P_L[:, :self.embed_dim]  # 截断
            elif P_L.shape[-1] < self.embed_dim:
                # 填充零
                padding = torch.zeros(P_L.shape[0], self.embed_dim - P_L.shape[-1], device=P_L.device, dtype=P_L.dtype)
                P_L_processed = torch.cat([P_L, padding], dim=-1)
            else:
                P_L_processed = P_L
            
            self.external_cls_token = P_L_processed.unsqueeze(1)  # [N, 1, embed_dim]
            
            # 根据LLM embedding生成语义部分信息
            self._generate_semantic_parts(P_L_processed)
        else:
            self.external_cls_token = None
            self.semantic_parts_embedding = None
    
    def _generate_semantic_parts(self, P_L):
        """
        根据LLM embedding生成语义部分信息
        Args:
            P_L: [N, embed_dim] 处理后的LLM embedding
        """
        # 如果num_semantic_parts为0，则不生成语义部分信息
        if self.num_semantic_parts == 0:
            self.semantic_parts_embedding = None
            return
            
        batch_size = P_L.shape[0]
        device = P_L.device
        
        # 将LLM embedding划分为num_semantic_parts个部分
        part_dim = self.embed_dim // self.num_semantic_parts
        
        # 生成语义部分的权重分布
        semantic_weights = []
        for i in range(self.num_semantic_parts):
            start_idx = i * part_dim
            end_idx = min((i + 1) * part_dim, self.embed_dim)
            part_embedding = P_L[:, start_idx:end_idx]  # [N, part_dim]
            
            # 计算该部分的重要性权重（使用L2范数）
            part_weight = torch.norm(part_embedding, p=2, dim=-1, keepdim=True)  # [N, 1]
            semantic_weights.append(part_weight)
        
        # 将权重堆叠并归一化
        semantic_weights = torch.cat(semantic_weights, dim=-1)  # [N, num_semantic_parts]
        semantic_weights = F.softmax(semantic_weights, dim=-1)  # 归一化为概率分布
        
        self.semantic_parts_embedding = semantic_weights  # [N, num_semantic_parts]
    
    def get_cls_token(self, batch_size, device):
        """
        获取当前使用的 cls_token
        Returns:
            [N, 1, embed_dim] 的 cls_token
        """
        if self.external_cls_token is not None:
            # 使用外部设置的 LLM embedding 作为 cls_token
            return self.external_cls_token + self.pos_embed[:, :1, :].to(device)
        else:
            # 使用默认的可学习 cls_token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            return cls_token.expand(batch_size, -1, -1)
    
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
    
    def semantic_aware_masking(self, x, mask_ratio, mask_lambda=1.0, hometown_len_list=None, 
                              destination_start_list=None, destination_end_list=None, valid_mask=None):
        """
        实现语义感知的混合masking策略
        Args:
            x: [N, L, D] 输入序列
            mask_ratio: 掩码比例
            mask_lambda: 混合策略权重参数
            hometown_len_list, destination_start_list, destination_end_list: 序列位置信息
            valid_mask: 有效位置掩码
        Returns:
            x_masked, mask, ids_keep_list, ids_restore_list, keep_mask
        """
        N, L, D = x.shape
        
        if valid_mask is None:
            valid_mask = torch.ones(N, L, dtype=torch.bool, device=x.device)
        
        # 如果num_semantic_parts为0或没有语义部分信息，回退到原始masking策略
        if self.num_semantic_parts == 0 or self.semantic_parts_embedding is None:
            return self.random_masking(x, mask_ratio, hometown_len_list, 
                                     destination_start_list, destination_end_list, valid_mask)
        
        # 获取语义权重 [N, num_semantic_parts]
        semantic_weights = self.semantic_parts_embedding
        
        final_masks = []
        ids_keep_list = []
        ids_restore_list = []
        x_masked_list = []
        
        for i in range(N):
            # 获取当前样本的有效位置
            valid_positions = torch.nonzero(valid_mask[i], as_tuple=True)[0]
            
            if len(valid_positions) == 0:
                # 没有有效位置，保持原序列
                ids_keep = torch.arange(L, device=x.device)
                mask = torch.zeros(L, dtype=torch.bool, device=x.device)
                x_masked = x[i]
            else:
                # 根据语义权重分配序列位置到不同语义部分
                semantic_assignment = self._assign_positions_to_semantic_parts(
                    valid_positions, semantic_weights[i])
                
                # 计算每个语义部分的mask数量
                mask_per_part = self._calculate_semantic_mask_counts(
                    semantic_assignment, mask_ratio, mask_lambda, L)
                
                # 生成最终的mask
                mask = self._generate_semantic_mask(
                    L, semantic_assignment, mask_per_part, valid_positions)
                
                # 获取保留的位置
                keep_indices = torch.nonzero(~mask, as_tuple=True)[0]
                ids_keep = keep_indices
                x_masked = x[i, ids_keep, :]
            
            # 创建restore索引
            ids_restore = torch.argsort(ids_keep)
            
            final_masks.append(mask)
            ids_keep_list.append(ids_keep)
            ids_restore_list.append(ids_restore)
            x_masked_list.append(x_masked)
        
        # 填充到相同长度
        max_keep = max(x_m.shape[0] for x_m in x_masked_list)
        x_masked_padded = torch.zeros(N, max_keep, D, device=x.device)
        keep_mask = torch.zeros(N, max_keep, dtype=torch.bool, device=x.device)
        
        for i, x_m in enumerate(x_masked_list):
            length = x_m.shape[0]
            x_masked_padded[i, :length] = x_m
            keep_mask[i, :length] = True
        
        mask = torch.stack(final_masks, dim=0)
        
        return x_masked_padded, mask, ids_keep_list, ids_restore_list, keep_mask
    
    def _assign_positions_to_semantic_parts(self, valid_positions, semantic_weights):
        """
        根据语义权重将序列位置分配到不同的语义部分
        Args:
            valid_positions: 有效位置索引
            semantic_weights: [num_semantic_parts] 语义权重
        Returns:
            semantic_assignment: dict, {part_id: [position_indices]}
        """
        num_positions = len(valid_positions)
        device = valid_positions.device
        
        # 根据权重计算每个部分应该分配的位置数量
        positions_per_part = (semantic_weights * num_positions).round().int()
        
        # 确保总数不超过可用位置数
        while positions_per_part.sum() > num_positions:
            max_idx = positions_per_part.argmax()
            positions_per_part[max_idx] -= 1
        
        # 如果总数不足，补充到权重最大的部分
        while positions_per_part.sum() < num_positions:
            max_idx = semantic_weights.argmax()
            positions_per_part[max_idx] += 1
        
        # 随机分配位置到各个部分
        shuffled_positions = valid_positions[torch.randperm(num_positions, device=device)]
        
        semantic_assignment = {}
        start_idx = 0
        for part_id in range(self.num_semantic_parts):
            count = positions_per_part[part_id].item()
            if count > 0:
                semantic_assignment[part_id] = shuffled_positions[start_idx:start_idx + count]
                start_idx += count
            else:
                semantic_assignment[part_id] = torch.tensor([], dtype=torch.long, device=device)
        
        return semantic_assignment
    
    def _calculate_semantic_mask_counts(self, semantic_assignment, mask_ratio, mask_lambda, total_length):
        """
        计算每个语义部分的mask数量，实现混合masking策略
        Args:
            semantic_assignment: 语义部分分配
            mask_ratio: 总体mask比例
            mask_lambda: 混合权重
            total_length: 序列总长度
        Returns:
            mask_counts: dict, {part_id: mask_count}
        """
        # 策略1：均匀masking - 每个部分按相同比例mask
        mask_counts_uniform = {}
        for part_id, positions in semantic_assignment.items():
            if len(positions) > 0:
                mask_counts_uniform[part_id] = int(len(positions) * mask_ratio)
            else:
                mask_counts_uniform[part_id] = 0
        
        # 策略2：平衡masking - 考虑全局约束
        total_positions = sum(len(positions) for positions in semantic_assignment.values())
        target_total_masks = int(total_positions * mask_ratio)
        
        # 随机排序语义部分
        part_ids = list(range(self.num_semantic_parts))
        random.shuffle(part_ids)
        
        mask_counts_balanced = {}
        cumulative_masks = 0
        
        for part_id in part_ids:
            positions = semantic_assignment[part_id]
            if len(positions) == 0:
                mask_counts_balanced[part_id] = 0
                continue
            
            remaining_masks = target_total_masks - cumulative_masks
            max_possible = len(positions)
            
            if remaining_masks <= 0:
                mask_counts_balanced[part_id] = 0
            else:
                mask_count = min(max_possible, remaining_masks)
                mask_counts_balanced[part_id] = mask_count
                cumulative_masks += mask_count
        
        # 混合两种策略
        final_mask_counts = {}
        for part_id in range(self.num_semantic_parts):
            count1 = mask_counts_uniform.get(part_id, 0)
            count2 = mask_counts_balanced.get(part_id, 0)
            final_count = int(mask_lambda * count1 + (1 - mask_lambda) * count2)
            final_mask_counts[part_id] = final_count
        
        return final_mask_counts
    
    def _generate_semantic_mask(self, seq_length, semantic_assignment, mask_counts, valid_positions):
        """
        根据语义分配和mask计数生成最终的mask
        Args:
            seq_length: 序列长度
            semantic_assignment: 语义部分分配
            mask_counts: 每个部分的mask数量
            valid_positions: 有效位置
        Returns:
            mask: [seq_length] bool tensor
        """
        device = valid_positions.device
        mask = torch.zeros(seq_length, dtype=torch.bool, device=device)
        
        for part_id, positions in semantic_assignment.items():
            mask_count = mask_counts.get(part_id, 0)
            if len(positions) > 0 and mask_count > 0:
                # 在该部分的位置中随机选择要mask的位置
                mask_count = min(mask_count, len(positions))
                if mask_count > 0:
                    indices_to_mask = torch.randperm(len(positions), device=device)[:mask_count]
                    positions_to_mask = positions[indices_to_mask]
                    mask[positions_to_mask] = True
        
        return mask
    
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
    
    def forward_encoder(self, x, mask_ratio=0.75, mask_lambda=1.0, hometown_len_list=None, destination_start_list=None, 
                       destination_end_list=None, valid_mask=None, training=True, use_semantic_masking=True):
        """
        Forward through encoder
        x: [N, L, D]
        mask_lambda: 语义感知masking的混合权重参数
        hometown_len_list: list of int, length of hometown sequence for each sample
        destination_start_list: list of int, start position of destination sequence
        destination_end_list: list of int, end position of destination sequence
        valid_mask: [N, L], True for valid positions
        use_semantic_masking: 是否使用语义感知masking策略
        """
        # Add pos embed (without cls token)
        if x.shape[1] <= self.pos_embed.shape[1] - 1:  # -1 for cls token
            x = x + self.pos_embed[:, 1:x.shape[1] + 1, :]  # Skip cls token pos embed
        else:
            # Handle sequences longer than expected
            pos_embed_extended = self.pos_embed[:, 1:, :].repeat(1, (x.shape[1] // (self.pos_embed.shape[1] - 1)) + 1, 1)
            x = x + pos_embed_extended[:, :x.shape[1], :]
        
        # Masking - 根据参数选择masking策略
        if training:
            if use_semantic_masking and self.num_semantic_parts > 0 and self.semantic_parts_embedding is not None:
                # 使用语义感知masking策略
                x_masked, mask, ids_keep_list, ids_restore_list, keep_mask = self.semantic_aware_masking(
                    x, mask_ratio, mask_lambda, hometown_len_list, destination_start_list, destination_end_list, valid_mask
                )
            else:
                # 使用原始随机masking策略
                x_masked, mask, ids_keep_list, ids_restore_list, keep_mask = self.random_masking(
                    x, mask_ratio, hometown_len_list, destination_start_list, destination_end_list, valid_mask
                )
        else:
            x_masked, mask, ids_keep_list, ids_restore_list, keep_mask = self.fixed_masking(
                x, hometown_len_list, destination_start_list, destination_end_list, valid_mask
            )
        
        # 获取 cls_token（可能是默认的或外部设置的LLM embedding）
        cls_tokens = self.get_cls_token(x_masked.shape[0], x.device)
        
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
    
    def forward(self, x, mask_ratio=0.75, mask_lambda=1.0, hometown_len_list=None, destination_start_list=None, 
               destination_end_list=None, valid_mask=None, training=True, use_semantic_masking=True):
        """
        Forward pass
        x: [N, L, D] input sequence (hometown + destination concatenated)
        mask_lambda: 语义感知masking的混合权重参数 (0.0-1.0)
        hometown_len_list: list of int, length of hometown sequence for each sample
        destination_start_list: list of int, start position of destination sequence
        destination_end_list: list of int, end position of destination sequence
        valid_mask: [N, L] mask indicating valid positions (True for valid, False for padding)
        use_semantic_masking: 是否使用语义感知masking策略
        
        Note: 
        1. 使用 set_external_cls_token() 方法来设置 LLM embedding 作为 cls_token，
           这会自动生成语义部分信息用于语义感知masking
        2. 当 num_semantic_parts = 0 时，即使设置了LLM embedding，也只使用随机masking策略
        3. 当 use_semantic_masking = False 时，强制使用随机masking策略
        """
        original_length = x.shape[1]
        
        # Encoder
        latent, mask, ids_keep_list, ids_restore_list, keep_mask = self.forward_encoder(
            x, mask_ratio, mask_lambda, hometown_len_list, destination_start_list, destination_end_list, valid_mask, training, use_semantic_masking
        )
        
        # Decoder
        pred = self.forward_decoder(latent, ids_keep_list, ids_restore_list, keep_mask, original_length)
        
        # Loss (only during training)
        if training:
            loss = self.forward_loss(x, pred, mask, valid_mask)
            return loss, pred, mask
        else:
            return pred, mask
