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
                 lambda_diversity=0.1, lambda_attn_reg=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_semantic_parts = num_semantic_parts  # Number of semantic segments
        self.lambda_diversity = lambda_diversity  # Weight for semantic diversity loss
        # Attention regularization (entropy regularization + optional max-weight penalty)
        self.lambda_attn_reg = lambda_attn_reg
        self.lambda_attn_max = 0.0
        # Cache attention weights during training for regularization
        self._last_attention_weights = None
        self._last_attention_valid_mask = None

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Token used to rebuild the full sequence on the encoder side for inspection only
        self.encoder_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.external_cls_token = None  # Stores external LLM embeddings as the cls token
        self.semantic_parts_embedding = None  # Stores semantic component representations
        self.pos_embed = nn.Parameter(
            torch.zeros(1, seq_len + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding, +1 for cls token

        # Semantic encoding network for generating num_semantic_parts representations
        # F_p = F_c ◦ sigmoid(W_{c2} tanh(W_{c1} F_c))
        if num_semantic_parts > 0:
            self.semantic_W_c1_list = nn.ModuleList([
                nn.Linear(embed_dim, embed_dim) for _ in range(num_semantic_parts)
            ])
            self.semantic_W_c2_list = nn.ModuleList([
                nn.Linear(embed_dim, embed_dim) for _ in range(num_semantic_parts)
            ])
            # Linear layer that fuses semantic representations with encoder outputs
            self.semantic_fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
            # Linear layer that produces attention weights
            self.semantic_attention_layer = nn.Linear(embed_dim, 1)

            # Cross-attention projection layers (Q from encoder_output, K/V from semantic_parts_embedding)
            self.semantic_q = nn.Linear(embed_dim, embed_dim, bias=True)
            self.semantic_k = nn.Linear(embed_dim, embed_dim, bias=True)
            self.semantic_v = nn.Linear(embed_dim, embed_dim, bias=True)
            self.semantic_out = nn.Linear(embed_dim, embed_dim, bias=True)

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
            nn.TransformerEncoderLayer(  # Reuse encoder blocks as the decoder
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
        Assign an external LLM embedding as the cls token and generate semantic components.
        Args:
            P_L: [N, D] LLM embedding. Truncate if D > embed_dim, pad with zeros if D < embed_dim.

        Note: When num_semantic_parts = 0, only the cls token is set and no semantic components are produced;
              the model falls back to standard random masking in that case.
        """
        if P_L is not None:
            # Handle potential dimensionality mismatches
            if P_L.shape[-1] > self.embed_dim:
                P_L_processed = P_L[:, :self.embed_dim]  # Truncate
            elif P_L.shape[-1] < self.embed_dim:
                # Zero-pad shorter embeddings
                padding = torch.zeros(P_L.shape[0], self.embed_dim - P_L.shape[-1], device=P_L.device, dtype=P_L.dtype)
                P_L_processed = torch.cat([P_L, padding], dim=-1)
            else:
                P_L_processed = P_L
            
            self.external_cls_token = P_L_processed.unsqueeze(1)  # [N, 1, embed_dim]
            
            # Derive semantic component representations from the LLM embedding
            self._generate_semantic_parts(P_L_processed)
        else:
            self.external_cls_token = None
            self.semantic_parts_embedding = None
    
    def _generate_semantic_parts(self, P_L):
        """
        Generate semantic component representations from the LLM embedding using
        F_p = F_c ◦ sigmoid(W_{c2} tanh(W_{c1} F_c)).
        Args:
            P_L: [N, embed_dim] processed LLM embedding (F_c)
        """
        # Skip semantic component construction if num_semantic_parts is zero
        if self.num_semantic_parts == 0:
            self.semantic_parts_embedding = None
            return
            
        batch_size = P_L.shape[0]
        device = P_L.device
        
        # Generate num_semantic_parts semantic representations
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
            F_p = P_L * gate  # [N, embed_dim] element-wise multiplication
            
            semantic_parts_list.append(F_p)
        
        # Stack all semantic representations: [N, num_semantic_parts, embed_dim]
        self.semantic_parts_embedding = torch.stack(semantic_parts_list, dim=1)
    
    def apply_semantic_encoding(self, encoder_output, return_attention_weights=False):
        """
        Fuse semantic representations with encoder outputs via cross-attention.
        - Semantic part embeddings ([N, P, D]) act as keys and values.
        - Encoder outputs ([N, L, D]) act as queries.
        Returns:
            semantic_enhanced: [N, L, D]
            attention_weights (optional): [N, P, L, 1] attention distribution over semantic parts
        """
        if self.num_semantic_parts == 0 or self.semantic_parts_embedding is None:
            if return_attention_weights:
                return None, None
            else:
                return None

        N, L, D = encoder_output.shape
        P = self.num_semantic_parts

        # Linear projections into the Q/K/V spaces
        # Q: [N, L, D], K: [N, P, D], V: [N, P, D]
        Q = self.semantic_q(encoder_output)
        K = self.semantic_k(self.semantic_parts_embedding)
        V = self.semantic_v(self.semantic_parts_embedding)

        # Attention scores: Q @ K^T / sqrt(D)
        # QK^T: [N, L, P]
        scale = D ** 0.5
        attn_logits = torch.matmul(Q, K.transpose(1, 2)) / scale  # [N, L, P]
        attn = torch.softmax(attn_logits, dim=-1)  # Softmax over the semantic dimension

        # Reformat attention weights for inspection/regularization: [N, P, L, 1]
        attn_for_return = attn.permute(0, 2, 1).unsqueeze(-1)  # [N, P, L, 1]

        # Weighted sum yields the cross-attended representations: [N, L, D]
        semantic_context = torch.matmul(attn, V)  # [N, L, D]
        semantic_enhanced = self.semantic_out(semantic_context) + encoder_output  # Residual connection

        if return_attention_weights:
            return semantic_enhanced, attn_for_return
        else:
            return semantic_enhanced
    
    def get_cls_token(self, batch_size, device):
        """
        Retrieve the cls token to use for the current batch.
        Returns:
            [N, 1, embed_dim] cls token tensor
        """
        if self.external_cls_token is not None:
            # Use the externally provided LLM embedding as the cls token
            return self.external_cls_token + self.pos_embed[:, :1, :].to(device)
        else:
            # Fall back to the default learnable cls token
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

        # For semantic projection layers (W_c1/W_c2), use a slightly larger random init variance
        if self.num_semantic_parts > 0:
            std = 0.1  # larger than default 0.02 to add variability
            for lin in self.semantic_W_c1_list:
                nn.init.normal_(lin.weight, mean=0.0, std=std)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
            for lin in self.semantic_W_c2_list:
                nn.init.normal_(lin.weight, mean=0.0, std=std)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
    
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
        Legacy semantic-aware masking strategy retained for backward compatibility.
        Note: The new semantic encoding pipeline no longer relies on this masking scheme.
        """
        N, L, D = x.shape
        
        if valid_mask is None:
            valid_mask = torch.ones(N, L, dtype=torch.bool, device=x.device)
        
        # Fall back to the original random masking when no semantic part information is available
        if self.num_semantic_parts == 0 or self.semantic_parts_embedding is None:
            return self.random_masking(x, mask_ratio, hometown_len_list, 
                                     destination_start_list, destination_end_list, valid_mask)
        
        # The current semantic encoding pipeline simply uses random masking
        return self.random_masking(x, mask_ratio, hometown_len_list, 
                                 destination_start_list, destination_end_list, valid_mask)
    
    def random_masking(self, x, mask_ratio, hometown_len_list=None, destination_start_list=None, destination_end_list=None, valid_mask=None):
        """
        Perform per-sample random masking by per-sample shuffling for destination sequences.
        During training we may mask the entire destination sequence, including the first and last elements.
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
        mask_lambda: blending factor for legacy semantic-aware masking (deprecated)
        hometown_len_list: list of int, length of hometown sequence for each sample
        destination_start_list: list of int, start position of destination sequence
        destination_end_list: list of int, end position of destination sequence
        valid_mask: [N, L], True for valid positions
        use_semantic_masking: toggle for legacy semantic-aware masking (deprecated)

        Note: Semantic enrichment now happens in the decoder; the encoder always applies random masking.
        """
        # Add pos embed (without cls token)
        if x.shape[1] <= self.pos_embed.shape[1] - 1:  # -1 for cls token
            x = x + self.pos_embed[:, 1:x.shape[1] + 1, :]  # Skip cls token pos embed
        else:
            # Handle sequences longer than expected
            pos_embed_extended = self.pos_embed[:, 1:, :].repeat(1, (x.shape[1] // (self.pos_embed.shape[1] - 1)) + 1, 1)
            x = x + pos_embed_extended[:, :x.shape[1], :]
        
        # Masking step – the new semantic encoding pipeline always uses random masking
        if training:
            # Apply random masking; semantic processing now happens inside the decoder
            x_masked, mask, ids_keep_list, ids_restore_list, keep_mask = self.random_masking(
                x, mask_ratio, hometown_len_list, destination_start_list, destination_end_list, valid_mask
            )
        else:
            x_masked, mask, ids_keep_list, ids_restore_list, keep_mask = self.fixed_masking(
                x, hometown_len_list, destination_start_list, destination_end_list, valid_mask
            )
        
        # Fetch the appropriate cls token (either default or external LLM-based)
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
        
        # Apply semantic enhancement when semantic encoding is enabled
        if self.num_semantic_parts > 0 and self.semantic_parts_embedding is not None:
            # Remove the cls token before semantic enhancement
            x_no_cls = x_encoded[:, 1:, :]  # [N, L, embed_dim]
            
            # Apply semantic encoding; add position embeddings beforehand to keep location-specific semantics.
            # During inference, rebuild the full encoder sequence before semantic fusion for inspection.
            if not training:
                # 1) Reconstruct the full-length encoder sequence (filling masked positions with encoder_mask_token)
                D_enc = x_encoded.shape[-1]
                full_encoded = self.encoder_mask_token.repeat(N, original_length, 1).to(x_encoded.device)  # [N, L, D_enc]
                
                for i in range(N):
                    # Number of valid encoder tokens (excluding cls)
                    valid_len = int(keep_mask[i].sum().item() - 1)
                    if valid_len > 0:
                        ids_keep = ids_keep_list[i]
                        # The first valid_len entries after removing cls correspond to the kept tokens
                        full_encoded[i, ids_keep, :] = x_encoded[i, 1:1 + valid_len, :]
                
                # 2) Add decoder positional embeddings (when dimensions match) and compute semantic attention
                if full_encoded.shape[-1] == self.decoder_pos_embed.shape[-1]:
                    pos_full = self.decoder_pos_embed[:, 1:original_length + 1, :].to(full_encoded.device)
                    full_encoded = full_encoded + pos_full
                _, attention_weights = self.apply_semantic_encoding(full_encoded, return_attention_weights=True)
                
                # 3) Log the most influential semantic component for each user's destination sequence
                if attention_weights is not None and uid is not None and destination_start_list is not None and destination_end_list is not None:
                    # attention_weights: [N, num_semantic_parts, L, 1]
                    max_semantic_indices_full = torch.argmax(attention_weights.squeeze(-1), dim=1)  # [N, L]
                    for i in range(N):
                        user_id = uid[i] if isinstance(uid[i], (int, str)) else uid[i].item()
                        dest_start = int(destination_start_list[i])
                        dest_end = int(destination_end_list[i])
                        if dest_end > dest_start:
                            dest_semantic_indices = max_semantic_indices_full[i, dest_start:dest_end + 1]
                            dest_semantic_list = dest_semantic_indices.detach().cpu().tolist()
                            # Summarize the distribution using plain Python integers for readability
                            vals, cnts = torch.unique(dest_semantic_indices, return_counts=True)
                            dist = {int(v.item()): int(c.item()) for v, c in zip(vals, cnts)}
                            print(f"The semantic representation number of the user {user_id}'s destination sequence: {dest_semantic_list}")
                            print(f"  - Length: {dest_end - dest_start + 1}")
                            print(f"  - Semantic representation distribution: {dist}")
                
                # 4) Add decoder positional embeddings (if compatible) to the kept tokens and run semantic fusion
                x_no_cls_for_sem = x_no_cls
                if x_no_cls.shape[-1] == self.decoder_pos_embed.shape[-1]:
                    L_keep = x_no_cls.shape[1]
                    pos_keep = self.decoder_pos_embed[:, 1:L_keep + 1, :].to(x_no_cls.device)
                    x_no_cls_for_sem = x_no_cls + pos_keep
                semantic_enhanced = self.apply_semantic_encoding(x_no_cls_for_sem, return_attention_weights=False)
            else:
                # During training, fuse semantics on kept tokens and cache attention for regularization
                x_no_cls_for_sem = x_no_cls
                if x_no_cls.shape[-1] == self.decoder_pos_embed.shape[-1]:
                    L_keep = x_no_cls.shape[1]
                    pos_keep = self.decoder_pos_embed[:, 1:L_keep + 1, :].to(x_no_cls.device)
                    x_no_cls_for_sem = x_no_cls + pos_keep
                semantic_enhanced, attention_weights = self.apply_semantic_encoding(
                    x_no_cls_for_sem, return_attention_weights=True
                )
                self._last_attention_weights = attention_weights  # [N, P, L_keep, 1]
                self._last_attention_valid_mask = keep_mask[:, 1:]  # [N, L_keep]
            
            if semantic_enhanced is not None:
                # Decode once by concatenating the semantic-enhanced tokens with the CLS token
                cls_tokens = x_encoded[:, :1, :]  # [N, 1, embed_dim]
                current_with_cls = torch.cat([cls_tokens, semantic_enhanced], dim=1)
                final_output = self._decode_single_semantic(
                    current_with_cls, ids_keep_list, ids_restore_list, keep_mask, original_length
                )
                return final_output

        # If semantic encoding is disabled, fall back to the vanilla decoder
        # Clear the cached attention values to avoid reusing outdated information
        self._last_attention_weights = None
        self._last_attention_valid_mask = None
        return self._decode_single_semantic(x_encoded, ids_keep_list, ids_restore_list, keep_mask, original_length)
    
    def _decode_single_semantic(self, x_encoded, ids_keep_list, ids_restore_list, keep_mask, original_length):
        """
        Decode a single semantic representation using the original decoder flow.
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
                # Fill masked positions with the decoder-embedded CLS token
                cls_dec = x[i, :1, :]  # [1, D]
                mask_tokens = cls_dec.repeat(num_mask_tokens, 1)  # [num_mask, D]
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
        
        # Positional encodings were already applied during semantic fusion; no extra decoder_pos_embed needed here
        
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

        # Compute MSE loss per position
        mse_loss = F.mse_loss(pred, original, reduction='none')  # [N, L, D]
        mse_loss = mse_loss.mean(dim=-1)  # [N, L]

        # Reconstruction loss over masked valid positions
        reconstruction_loss = (mse_loss * loss_mask.float()).sum() / loss_mask.sum()

        # Semantic diversity loss (if enabled)
        diversity_loss = self.compute_semantic_diversity_loss()

        # Attention distribution regularization (training caches it)
        attn_reg_loss = self.compute_attention_reg_loss()

        # Total loss
        total_loss = reconstruction_loss + diversity_loss + attn_reg_loss

        return total_loss
    
    def forward_loss_detailed(self, original, pred, mask, valid_mask=None):
        """
        Provide a detailed breakdown of the loss components.
        Args:
            original: [N, L, D] original input
            pred: [N, L, D] reconstructed output
            mask: [N, L] mask over positions
            valid_mask: [N, L] validity mask
        Returns:
            dict: dictionary containing each loss component
        """
        if valid_mask is None:
            valid_mask = torch.ones_like(mask, dtype=torch.bool)
        
        # Only compute loss on masked AND valid positions
        loss_mask = mask & valid_mask  # [N, L]

        if loss_mask.sum() == 0:
            return {
                'total_loss': torch.tensor(0.0, device=original.device),
                'reconstruction_loss': torch.tensor(0.0, device=original.device),
                'diversity_loss': torch.tensor(0.0, device=original.device),
                'diversity_weight': self.lambda_diversity,
                'attention_reg_loss': torch.tensor(0.0, device=original.device),
                'attention_reg_weight': self.lambda_attn_reg,
                'attention_max_weight': self.lambda_attn_max
            }
        
        # Compute MSE loss
        mse_loss = F.mse_loss(pred, original, reduction='none')  # [N, L, D]
        mse_loss = mse_loss.mean(dim=-1)  # [N, L], mean loss per position
        
        # Apply mask and compute mean loss on masked positions
        reconstruction_loss = (mse_loss * loss_mask.float()).sum() / loss_mask.sum()
        
        # Compute semantic diversity penalty when semantic encoding is active
        diversity_loss = self.compute_semantic_diversity_loss()

        # Attention distribution regularization (only populated during training)
        attn_reg_loss = self.compute_attention_reg_loss()
        
        # Total loss
        total_loss = reconstruction_loss + diversity_loss + attn_reg_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'diversity_loss': diversity_loss,
            'diversity_weight': self.lambda_diversity,
            'attention_reg_loss': attn_reg_loss,
            'attention_reg_weight': self.lambda_attn_reg,
            'attention_max_weight': self.lambda_attn_max
        }
    
    def compute_semantic_diversity_loss(self):
        """
        Compute the semantic diversity loss by aggregating pairwise cosine similarities
        between distinct semantic representations.
        Returns:
            diversity_loss: scalar diversity penalty
        """
        # Skip the diversity penalty when semantic encoding is disabled or has fewer than two parts
        if (self.num_semantic_parts <= 1 or 
            self.semantic_parts_embedding is None):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # semantic_parts_embedding: [N, num_semantic_parts, embed_dim]
        N, num_parts, embed_dim = self.semantic_parts_embedding.shape
        
        # Normalize semantic representations before cosine similarity
        semantic_normalized = F.normalize(self.semantic_parts_embedding, p=2, dim=-1)  # [N, num_parts, embed_dim]
        
        # Pairwise cosine similarity via batched matrix multiplication
        # [N, num_parts, embed_dim] @ [N, embed_dim, num_parts] = [N, num_parts, num_parts]
        cosine_matrix = torch.bmm(semantic_normalized, semantic_normalized.transpose(1, 2))  # [N, num_parts, num_parts]
        
        # Upper-triangular mask excludes self-similarity terms
        mask = torch.triu(torch.ones(num_parts, num_parts, device=cosine_matrix.device), diagonal=1).bool()
        
        # Retain only the upper-triangular elements to avoid double counting
        upper_triangle_similarities = cosine_matrix[:, mask]  # [N, num_pairs]
        
        # Average squared cosine similarity discourages degenerate negative correlations
        average_cosine_similarity = upper_triangle_similarities.pow(2).mean()
        
        # Multiply by the diversity weight so that lower similarity implies lower penalty
        diversity_loss = self.lambda_diversity * average_cosine_similarity
        
        return diversity_loss

    def compute_attention_reg_loss(self):
        """
        Regularize the semantic attention distribution via:
        - Entropy regularization to encourage balanced attention across semantic parts.
        - Optional maximum-weight penalty to prevent dominance of a single part.
        Returns: scalar loss value (weights applied internally).
        """
        if self._last_attention_weights is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        attn = self._last_attention_weights  # [N, P, L, 1]
        attn = attn.squeeze(-1)  # [N, P, L]

        if self._last_attention_valid_mask is None:
            valid_mask = torch.ones(attn.shape[0], attn.shape[-1], dtype=torch.bool, device=attn.device)
        else:
            valid_mask = self._last_attention_valid_mask  # [N, L]

        # Entropy regularization across the semantic dimension
        eps = 1e-8
        # Entropy: -sum p log p; normalize by log(P) so the maximum entropy equals 1
        entropy = -(attn * (attn + eps).log()).sum(dim=1)  # [N, L]
        P = attn.shape[1]
        max_entropy = torch.log(torch.tensor(float(P), device=attn.device))
        normalized_entropy = entropy / (max_entropy + eps)  # [N, L]

        # Average over valid temporal positions only
        valid_float = valid_mask.float()
        entropy_mean = (normalized_entropy * valid_float).sum() / (valid_float.sum() + eps)

        # Encourage high entropy: loss term becomes 1 - mean entropy
        entropy_reg = self.lambda_attn_reg * (1.0 - entropy_mean)

        # Optional penalty on maximum attention weights
        max_w = attn.max(dim=1).values  # [N, L]
        target = 1.0 / float(P)
        max_penalty = ((max_w - target).clamp(min=0.0) ** 2)
        max_penalty_mean = (max_penalty * valid_float).sum() / (valid_float.sum() + eps)
        max_reg = self.lambda_attn_max * max_penalty_mean

        return entropy_reg + max_reg
    
    def forward(self, x, uid, mask_ratio=0.75, mask_lambda=1.0, hometown_len_list=None, destination_start_list=None,
               destination_end_list=None, valid_mask=None, training=True, use_semantic_masking=True):
        """
        Forward pass with new semantic encoding system and diversity loss
        x: [N, L, D] input sequence (hometown + destination concatenated)
        mask_lambda: deprecated knob retained for backward compatibility
        hometown_len_list: list of int, length of hometown sequence for each sample
        destination_start_list: list of int, start position of destination sequence
        destination_end_list: list of int, end position of destination sequence
        valid_mask: [N, L] mask indicating valid positions (True for valid, False for padding)
        use_semantic_masking: deprecated toggle retained for backward compatibility

        Workflow of the semantic encoding system:
        1. Call set_external_cls_token() to provide the LLM embedding baseline (F_c).
        2. Generate num_semantic_parts semantic representations via F_p = F_c ◦ sigmoid(W_{c2} tanh(W_{c1} F_c)).
        3. The encoder applies standard random masking.
        4. The decoder processes each semantic representation and aggregates the outputs.

        Loss composition:
        - Reconstruction loss: mean squared error on masked positions.
        - Diversity loss: pairwise cosine similarity penalty encouraging diverse semantics.
        - Total = reconstruction loss + lambda_diversity * diversity loss + attention regularization.

        When num_semantic_parts = 0, the model reduces to a standard MAE (no semantic augmentation, no diversity loss).
        When num_semantic_parts <= 1, the diversity loss is skipped.
        """
        # Refresh semantic projections each forward pass to keep gradients flowing through projection layers
        if self.num_semantic_parts > 0 and self.external_cls_token is not None:
            # self.external_cls_token: [N, 1, D] -> [N, D]
            P_L_current = self.external_cls_token.squeeze(1)
            # Regenerate semantic parts based on the current batch of external CLS embeddings
            self._generate_semantic_parts(P_L_current)

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
