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
                 mlp_ratio=4., norm_layer=nn.LayerNorm, num_semantic_parts=8,
                 lambda_diversity=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_semantic_parts = num_semantic_parts  # 语义部分数量
        self.lambda_diversity = lambda_diversity  # 语义多样性损失权重

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 用于在Encoder侧重建完整序列时填充被mask的位置（仅用于语义打印，不参与反向传播）
        self.encoder_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.external_cls_token = None  # 用于存储外部LLM embedding作为cls_token
        self.semantic_parts_embedding = None  # 用于存储语义部分信息
        self.pos_embed = nn.Parameter(
            torch.zeros(1, seq_len + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding, +1 for cls token

        # 语义编码网络：用于生成 num_semantic_parts 个语义表征
        # F_p = F_c ◦ sigmoid(W_{c2} tanh(W_{c1} F_c))
        if num_semantic_parts > 0:
            self.semantic_W_c1_list = nn.ModuleList([
                nn.Linear(embed_dim, embed_dim) for _ in range(num_semantic_parts)
            ])
            self.semantic_W_c2_list = nn.ModuleList([
                nn.Linear(embed_dim, embed_dim) for _ in range(num_semantic_parts)
            ])
            # 用于将语义表征与encoder输出结合的线性层
            self.semantic_fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
            # 用于生成注意力权重的线性层
            self.semantic_attention_layer = nn.Linear(embed_dim, 1)

        self.encoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.encoder_norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, seq_len + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding, +1 for cls token

        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(  # 使用Encoder作为Decoder
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=int(decoder_embed_dim * mlp_ratio),
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, embed_dim, bias=True
        )  # decoder to original embedding
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
        使用新的语义编码方法：F_p = F_c ◦ sigmoid(W_{c2} tanh(W_{c1} F_c))
        Args:
            P_L: [N, embed_dim] 处理后的LLM embedding (F_c)
        """
        # 如果num_semantic_parts为0，则不生成语义部分信息
        if self.num_semantic_parts == 0:
            self.semantic_parts_embedding = None
            return
            
        batch_size = P_L.shape[0]
        device = P_L.device
        
        # 生成 num_semantic_parts 个语义表征
        # F_p = F_c ◦ sigmoid(W_{c2} tanh(W_{c1} F_c))
        semantic_parts_list = []
        
        for i in range(self.num_semantic_parts):
            # W_{c1} F_c
            h1 = self.semantic_W_c1_list[i](P_L)  # [N, embed_dim]
            # tanh(W_{c1} F_c)
            h1_tanh = torch.tanh(h1)  # [N, embed_dim]
            # W_{c2} tanh(W_{c1} F_c)
            h2 = self.semantic_W_c2_list[i](h1_tanh)  # [N, embed_dim]
            # sigmoid(W_{c2} tanh(W_{c1} F_c))
            gate = torch.sigmoid(h2)  # [N, embed_dim]
            # F_p = F_c ◦ sigmoid(W_{c2} tanh(W_{c1} F_c))
            F_p = P_L * gate  # [N, embed_dim] 元素级别乘法
            
            semantic_parts_list.append(F_p)
        
        # 将所有语义表征堆叠：[N, num_semantic_parts, embed_dim]
        self.semantic_parts_embedding = torch.stack(semantic_parts_list, dim=1)
    
    def apply_semantic_encoding(self, encoder_output, return_attention_weights=False):
        """
        将语义表征与encoder输出结合
        Args:
            encoder_output: [N, L, embed_dim] encoder的输出表征
            return_attention_weights: bool, 是否返回注意力权重信息
        Returns:
            semantic_enhanced_output: [N, num_semantic_parts, L, embed_dim] 语义增强的表征
            attention_weights: (optional) [N, num_semantic_parts, L, 1] 注意力权重
        """
        if self.num_semantic_parts == 0 or self.semantic_parts_embedding is None:
            if return_attention_weights:
                return None, None
            else:
                return None
            
        N, L, D = encoder_output.shape
        # semantic_parts_embedding: [N, num_semantic_parts, embed_dim]
        
        # 扩展语义表征到所有位置：[N, num_semantic_parts, L, embed_dim]
        semantic_expanded = self.semantic_parts_embedding.unsqueeze(2).expand(-1, -1, L, -1)
        
        # 扩展encoder输出到所有语义部分：[N, num_semantic_parts, L, embed_dim]
        encoder_expanded = encoder_output.unsqueeze(1).expand(-1, self.num_semantic_parts, -1, -1)
        
        # 拼接语义表征和encoder输出：[N, num_semantic_parts, L, 2*embed_dim]
        combined = torch.cat([semantic_expanded, encoder_expanded], dim=-1)
        
        # 通过线性层融合：[N, num_semantic_parts, L, embed_dim]
        fused = self.semantic_fusion_layer(combined)
        
        # 应用注意力机制生成权重
        attention_scores = self.semantic_attention_layer(fused)  # [N, num_semantic_parts, L, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # 在语义部分维度上softmax
        
        # 加权融合：[N, num_semantic_parts, L, embed_dim]
        semantic_enhanced = fused * attention_weights
        
        if return_attention_weights:
            return semantic_enhanced, attention_weights
        else:
            return semantic_enhanced
    
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
        pos_embed = self.get_1d_sincos_pos_embed(
            self.embed_dim, self.seq_len, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = self.get_1d_sincos_pos_embed(
            self.decoder_embed.out_features, self.seq_len, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize tokens
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.encoder_mask_token, std=0.02)

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
        注意：这个方法现在仅用于向后兼容，新的语义编码方法不需要这种masking策略
        """
        N, L, D = x.shape
        
        if valid_mask is None:
            valid_mask = torch.ones(N, L, dtype=torch.bool, device=x.device)
        
        # 如果num_semantic_parts为0或没有语义部分信息，回退到原始masking策略
        if self.num_semantic_parts == 0 or self.semantic_parts_embedding is None:
            return self.random_masking(x, mask_ratio, hometown_len_list, 
                                     destination_start_list, destination_end_list, valid_mask)
        
        # 新的语义编码方法不需要特殊的masking，直接使用随机masking
        return self.random_masking(x, mask_ratio, hometown_len_list, 
                                 destination_start_list, destination_end_list, valid_mask)
    
    def random_masking(self, x, mask_ratio, hometown_len_list=None, destination_start_list=None, destination_end_list=None, valid_mask=None):
        """
        Perform per-sample random masking by per-sample shuffling for destination sequences.
        Training时：可以mask整个destination sequence（包括头部和尾部位置）
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
                
                # Create mask - mask destination sequence (including start and end)
                mask = torch.zeros(L, dtype=torch.bool, device=x.device)
                
                if dest_end > dest_start:  # At least one position in destination sequence
                    # Maskable positions: all positions of destination sequence (including start and end)
                    maskable_positions = torch.arange(dest_start, dest_end, device=x.device)
                    
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
                    # Mask any valid positions (including first and last)
                    maskable_positions = valid_positions
                    
                    num_mask = int(len(maskable_positions) * mask_ratio)
                    mask = torch.zeros(L, dtype=torch.bool, device=x.device)
                    
                    if num_mask > 0:
                        noise = torch.rand(len(maskable_positions), device=x.device)
                        ids_shuffle = torch.argsort(noise)
                        mask_indices = maskable_positions[ids_shuffle[:num_mask]]
                        mask[mask_indices] = True
                    
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
        mask_lambda: 语义感知masking的混合权重参数（已弃用，新语义编码不使用特殊masking）
        hometown_len_list: list of int, length of hometown sequence for each sample
        destination_start_list: list of int, start position of destination sequence
        destination_end_list: list of int, end position of destination sequence
        valid_mask: [N, L], True for valid positions
        use_semantic_masking: 语义感知masking开关（已弃用，新语义编码在decoder中处理）
        
        注意：新的语义编码方法在decoder阶段进行语义增强，encoder阶段使用标准的随机masking
        """
        # Add pos embed (without cls token)
        if x.shape[1] <= self.pos_embed.shape[1] - 1:  # -1 for cls token
            x = x + self.pos_embed[:, 1:x.shape[1] + 1, :]  # Skip cls token pos embed
        else:
            # Handle sequences longer than expected
            pos_embed_extended = self.pos_embed[:, 1:, :].repeat(1, (x.shape[1] // (self.pos_embed.shape[1] - 1)) + 1, 1)
            x = x + pos_embed_extended[:, :x.shape[1], :]
        
        # Masking - 新语义编码方法统一使用随机masking
        if training:
            # 使用随机masking策略（新语义编码在decoder中处理）
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
    
    def forward_decoder(self, x_encoded, ids_keep_list, ids_restore_list, keep_mask, original_length, 
                      uid=None, destination_start_list=None, destination_end_list=None, training=True):
        """
        Forward through decoder with semantic enhancement
        Args:
            uid: list of user IDs for each sample in batch
            destination_start_list: list of destination start positions
            destination_end_list: list of destination end positions  
            training: whether in training mode
        """
        N = len(ids_keep_list)
        
        # 如果启用了语义编码，应用语义增强
        if self.num_semantic_parts > 0 and self.semantic_parts_embedding is not None:
            # 移除cls token进行语义增强
            x_no_cls = x_encoded[:, 1:, :]  # [N, L, embed_dim]
            
            # 应用语义编码：[N, num_semantic_parts, L, embed_dim]
            # 在推理时（非训练）先把mask的token添加回Encoder侧完整序列，再做语义嵌入并打印
            if not training:
                # 1) 基于Encoder输出重建完整长度的序列（被mask位置用encoder_mask_token填充）
                D_enc = x_encoded.shape[-1]
                full_encoded = self.encoder_mask_token.repeat(N, original_length, 1).to(x_encoded.device)  # [N, L, D_enc]
                
                for i in range(N):
                    # 有效的encoder token数量（不含cls）
                    valid_len = int(keep_mask[i].sum().item() - 1)
                    if valid_len > 0:
                        ids_keep = ids_keep_list[i]
                        # x_encoded中去掉cls后的前valid_len即为保留token的表示，顺序与ids_keep一致
                        full_encoded[i, ids_keep, :] = x_encoded[i, 1:1 + valid_len, :]
                
                # 2) 在完整序列上进行语义嵌入，并拿到注意力权重
                _, attention_weights = self.apply_semantic_encoding(full_encoded, return_attention_weights=True)
                
                # 3) 打印每个用户目的地序列的最大权重语义表征序号
                if attention_weights is not None and uid is not None and destination_start_list is not None and destination_end_list is not None:
                    # attention_weights: [N, num_semantic_parts, L, 1]
                    max_semantic_indices_full = torch.argmax(attention_weights.squeeze(-1), dim=1)  # [N, L]
                    for i in range(N):
                        user_id = uid[i] if isinstance(uid[i], (int, str)) else uid[i].item()
                        dest_start = int(destination_start_list[i])
                        dest_end = int(destination_end_list[i])
                        if dest_end > dest_start:
                            dest_semantic_indices = max_semantic_indices_full[i, dest_start:dest_end]
                            dest_semantic_list = dest_semantic_indices.detach().cpu().tolist()
                            # 统计分布（转为纯Python整数，避免打印设备信息）
                            vals, cnts = torch.unique(dest_semantic_indices, return_counts=True)
                            dist = {int(v.item()): int(c.item()) for v, c in zip(vals, cnts)}
                            print(f"[推理] 用户 {user_id} 的目的地序列语义表征序号: {dest_semantic_list}")
                            print(f"  - 目的地序列长度: {dest_end - dest_start}")
                            print(f"  - 语义表征分布: {dist}")
                
                # 4) 解码阶段仍按原策略：在保留token上做语义嵌入
                semantic_enhanced = self.apply_semantic_encoding(x_no_cls, return_attention_weights=False)
            else:
                # 训练阶段：保持原行为，仅在保留token上做语义嵌入，不打印
                semantic_enhanced = self.apply_semantic_encoding(x_no_cls, return_attention_weights=False)
            
            if semantic_enhanced is not None:
                # 对每个语义部分分别进行decoder处理
                semantic_outputs = []
                
                for part_idx in range(self.num_semantic_parts):
                    # 取出当前语义部分的表征：[N, L, embed_dim]
                    current_semantic = semantic_enhanced[:, part_idx, :, :]
                    
                    # 重新添加cls token
                    cls_tokens = x_encoded[:, :1, :]  # [N, 1, embed_dim]
                    current_with_cls = torch.cat([cls_tokens, current_semantic], dim=1)
                    
                    # 进行decoder处理
                    decoded_part = self._decode_single_semantic(
                        current_with_cls, ids_keep_list, ids_restore_list, keep_mask, original_length
                    )
                    semantic_outputs.append(decoded_part)
                
                # 归一化相加所有语义部分的输出：[N, L, embed_dim]
                final_output = torch.stack(semantic_outputs, dim=0).mean(dim=0)
                return final_output
        
        # 如果没有语义编码，使用原始decoder
        return self._decode_single_semantic(x_encoded, ids_keep_list, ids_restore_list, keep_mask, original_length)
    
    def _decode_single_semantic(self, x_encoded, ids_keep_list, ids_restore_list, keep_mask, original_length):
        """
        对单个语义表征进行decoder处理（原始decoder逻辑）
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
        mse_loss = F.mse_loss(pred, original, reduction='none')  # [N, L, D]
        mse_loss = mse_loss.mean(dim=-1)  # [N, L], mean loss per position
        
        # Apply mask and compute mean loss on masked positions
        reconstruction_loss = (mse_loss * loss_mask.float()).sum() / loss_mask.sum()
        
        # 计算语义多样性损失（如果启用了语义编码）
        diversity_loss = self.compute_semantic_diversity_loss()
        
        # 总损失
        total_loss = reconstruction_loss + diversity_loss
        
        return total_loss
    
    def forward_loss_detailed(self, original, pred, mask, valid_mask=None):
        """
        计算详细的损失信息，返回重建损失和多样性损失的分别值
        Args:
            original: [N, L, D] 原始输入
            pred: [N, L, D] 预测输出
            mask: [N, L] 掩码
            valid_mask: [N, L] 有效位置掩码
        Returns:
            dict: 包含各种损失的字典
        """
        if valid_mask is None:
            valid_mask = torch.ones_like(mask, dtype=torch.bool)
        
        # Only compute loss on masked AND valid positions
        loss_mask = mask & valid_mask  # [N, L]
        
        if loss_mask.sum() == 0:
            return {
                'total_loss': torch.tensor(0.0, device=original.device),
                'reconstruction_loss': torch.tensor(0.0, device=original.device),
                'diversity_loss': torch.tensor(0.0, device=original.device)
            }
        
        # Compute MSE loss
        mse_loss = F.mse_loss(pred, original, reduction='none')  # [N, L, D]
        mse_loss = mse_loss.mean(dim=-1)  # [N, L], mean loss per position
        
        # Apply mask and compute mean loss on masked positions
        reconstruction_loss = (mse_loss * loss_mask.float()).sum() / loss_mask.sum()
        
        # 计算语义多样性损失（如果启用了语义编码）
        diversity_loss = self.compute_semantic_diversity_loss()
        
        # 总损失
        total_loss = reconstruction_loss + diversity_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'diversity_loss': diversity_loss,
            'diversity_weight': self.lambda_diversity
        }
    
    def compute_semantic_diversity_loss(self):
        """
        计算语义表征多样性损失：所有互不相同表征的余弦相似度之和并归一化
        使用高效的矩阵运算实现
        Returns:
            diversity_loss: 多样性损失值
        """
        # 如果没有语义编码或语义部分数量少于2，则不计算多样性损失
        if (self.num_semantic_parts <= 1 or 
            self.semantic_parts_embedding is None):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # semantic_parts_embedding: [N, num_semantic_parts, embed_dim]
        N, num_parts, embed_dim = self.semantic_parts_embedding.shape
        
        # 归一化语义表征以计算余弦相似度
        semantic_normalized = F.normalize(self.semantic_parts_embedding, p=2, dim=-1)  # [N, num_parts, embed_dim]
        
        # 计算所有表征对之间的余弦相似度矩阵
        # 通过矩阵乘法计算：[N, num_parts, embed_dim] @ [N, embed_dim, num_parts] = [N, num_parts, num_parts]
        cosine_matrix = torch.bmm(semantic_normalized, semantic_normalized.transpose(1, 2))  # [N, num_parts, num_parts]
        
        # 创建上三角掩码，排除对角线元素（自己与自己的相似度）
        mask = torch.triu(torch.ones(num_parts, num_parts, device=cosine_matrix.device), diagonal=1).bool()
        
        # 只保留上三角部分（不包括对角线），避免重复计算
        upper_triangle_similarities = cosine_matrix[:, mask]  # [N, num_pairs]
        
        # 计算平均余弦相似度
        average_cosine_similarity = upper_triangle_similarities.mean()
        
        # 多样性损失：我们希望余弦相似度尽可能小（表征尽可能不同）
        # 因此损失为相似度的平均值，乘以权重系数
        diversity_loss = self.lambda_diversity * average_cosine_similarity
        
        return diversity_loss
    
    def forward(self, x, uid, mask_ratio=0.75, mask_lambda=1.0, hometown_len_list=None, destination_start_list=None,
               destination_end_list=None, valid_mask=None, training=True, use_semantic_masking=True):
        """
        Forward pass with new semantic encoding system and diversity loss
        x: [N, L, D] input sequence (hometown + destination concatenated)
        mask_lambda: 已弃用参数，保留为向后兼容
        hometown_len_list: list of int, length of hometown sequence for each sample
        destination_start_list: list of int, start position of destination sequence
        destination_end_list: list of int, end position of destination sequence
        valid_mask: [N, L] mask indicating valid positions (True for valid, False for padding)
        use_semantic_masking: 已弃用参数，保留为向后兼容
        
        新语义编码系统工作流程:
        1. 使用 set_external_cls_token() 设置 LLM embedding 作为语义编码基础 (F_c)
        2. 通过 F_p = F_c ◦ sigmoid(W_{c2} tanh(W_{c1} F_c)) 生成 num_semantic_parts 个语义表征
        3. Encoder 阶段使用标准随机masking
        4. Decoder 阶段对每个语义表征分别处理，最后归一化相加得到最终预测
        
        损失函数组成:
        - 重建损失: 标准的MSE损失，计算masked位置的重建误差
        - 多样性损失: 计算所有语义表征对之间的余弦相似度，鼓励表征多样性
        - 总损失 = 重建损失 + lambda_diversity * 多样性损失
        
        当 num_semantic_parts = 0 时，使用标准的MAE流程（无语义增强，无多样性损失）
        当 num_semantic_parts <= 1 时，不计算多样性损失
        """
        original_length = x.shape[1]
        
        # Encoder
        latent, mask, ids_keep_list, ids_restore_list, keep_mask = self.forward_encoder(
            x, mask_ratio, mask_lambda, hometown_len_list, destination_start_list, destination_end_list, valid_mask, training, use_semantic_masking
        )
        
        # Decoder
        pred = self.forward_decoder(latent, ids_keep_list, ids_restore_list, keep_mask, original_length,
                                  uid, destination_start_list, destination_end_list, training)
        
        # Loss (only during training)
        if training:
            loss = self.forward_loss(x, pred, mask, valid_mask)
            return loss, pred, mask
        else:
            return pred, mask
