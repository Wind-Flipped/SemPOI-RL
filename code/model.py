import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import repeat
from trainer import top_np_recommendation

from LLMs import TravelStyleGenerator, TravelStyleRewardCalculator
import numpy as np
from mae import MaskedAutoEncoder

class PositionalEncoding(nn.Module):
    """Sine positional encoding that injects location information into sequences."""
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
                norm_first=True,  # Use Pre-LN architecture for better stability
                activation=F.gelu,  # Explicitly set activation function
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

        # In PyTorch, src_key_padding_mask=True marks positions to ignore
        # Therefore invert d_pad (assuming True indicates a valid position)
        padding_mask = ~d_pad

        # Debug information and numerical-stability guardrails
        for i, layer in enumerate(self.transformer_stack):
            try:
                x_before = x.clone()
                x_new = layer(x, src_key_padding_mask=padding_mask)

                # Validate transformer output
                if torch.isnan(x_new).any() or torch.isinf(x_new).any():
                    print(f"Warning: NaN/Inf detected in transformer layer {i}")
                    print(f"Input range: [{x_before.min():.4f}, {x_before.max():.4f}]")
                    print(f"d_pad shape: {d_pad.shape}, unique values: {torch.unique(d_pad)}")
                    # Preserve the previous representation and skip this layer
                    continue
                else:
                    x = x_new

            except Exception as e:
                print(f"Error in transformer layer {i}: {e}")
                # Skip this layer and keep the prior representation
                continue

        x = x[:, -1, :]

        return x, self.gamma_proj(x), torch.nn.functional.softplus(self.tau_proj(x))  # Improves stability

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


# Construct total framework(AR-Trip)
class SemPOIModel(nn.Module):
    def __init__(self, args, poi_size, region_poi,
                 max_length_venue_id=100, max_length_ori_id=100, d_model=128, n_head=4, num_encoder_layers=1, n_tf_layers=4, d_z=128,):

        super(SemPOIModel, self).__init__()
        # initial LLMs
        self.travel_style_reward_calculator = TravelStyleRewardCalculator()
        if args.use_llm and not args.use_target_llm:
            self.travel_style_generator = TravelStyleGenerator(use_vllm=args.use_vllm, use_lora=args.use_lora, lora_path=args.lora_path, lora_path2=args.lora_path2)
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

        self.seq2seq_transformer = nn.Transformer(
            d_model=self.hidden_size,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,
            dim_feedforward=4 * self.hidden_size,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.seq_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.seq_norm = nn.LayerNorm(self.hidden_size)
        
        # Time and space embedding layers used by the ST module
        self.time_embedding = nn.Linear(1, self.hidden_size, bias=False)  # Time has one dimension
        self.space_embedding = nn.Linear(2, self.hidden_size, bias=False)  # Space has two dimensions

        # Positional encodings for the ST module
        self.src_pos_encoding = PositionalEncoding(self.hidden_size, dropout=0.1, max_len=max_length_ori_id + max_length_venue_id + 10)
        self.tgt_pos_encoding = PositionalEncoding(self.hidden_size, dropout=0.1, max_len=max_length_ori_id + max_length_venue_id + 10)

        # Project concatenated time + space + category features to a unified dimension
        self.concat_to_unified = nn.Linear(3 * self.hidden_size, self.hidden_size, bias=False)

        # Learnable mask token for the Masked AutoEncoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        nn.init.xavier_uniform_(self.mask_token)

        # Initialize MaskedAutoEncoder when ST module is enabled
        if self.args.st_module:
            max_seq_len = max_length_ori_id + max_length_venue_id
            # Respect the semantic partition count configured in main.py
            num_semantic_parts = args.num_semantic_parts  # Default 0 (disables semantic masking)
            self.mae = MaskedAutoEncoder(
                seq_len=max_seq_len,
                embed_dim=self.hidden_size,
                depth=6,
                num_heads=n_head,
                decoder_embed_dim=self.hidden_size,
                decoder_depth=4,
                decoder_num_heads=n_head,
                mlp_ratio=4.0,
                num_semantic_parts=num_semantic_parts,
                lambda_diversity=args.lambda_diversity,
                lambda_attn_reg=args.lambda_attn_reg,
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


    def forward(self, uid, messages, o_ck, query, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg, d_rg, target_seq=None):
        batch_size, seq_length = query.size()
        # region_repr = self.region_embedding.weight
        region_mask = torch.stack([self.region_masks[int(r)] for r in d_rg], dim=0)
        region_mask_uns = region_mask.unsqueeze(1).expand(-1, query.size(1), -1).to(query.device)
        # initialize
        query_emb = self.poi_embedding(query)  # [b,l,d]
        o_emb = self.poi_embedding(o_ck)
        d_target_emb = self.poi_embedding(d_ck)
        pad_vec = self.poi_embedding.emb.weight[0]

        if self.args.use_llm:
            # self.args.kg = False
            self.args.s_infer = False
            # Step 1: Generate travel-style text via the LLM
            if self.args.use_target_llm:
                generated_texts = messages
            else:
                generated_texts = self.travel_style_generator.get_output(messages, max_length=512, temperature=0.7)
            print("UID:", uid)
            print("Generated Texts:", generated_texts)
            # Step 2: Obtain embeddings for the generated text (truncate to self.hidden_size dimensions)
            generated_embeddings = self.travel_style_reward_calculator.get_embedding(generated_texts,
                                                                                     embedding_dim=self.llm_embedding_dim)
            generated_embeddings = torch.tensor(generated_embeddings).to(self.args.device)  # [b, d]
            # Apply L2 normalization to the LLM embeddings
            generated_embeddings = F.normalize(generated_embeddings, p=2, dim=-1)
            P_L = generated_embeddings.unsqueeze(1).expand([generated_embeddings.shape[0], d_target_emb.shape[1], generated_embeddings.shape[1]])
        

        if self.args.st_module:
            # Initialize loss accumulator
            seq2seq_loss = torch.tensor(0.0, device=self.args.device, dtype=torch.float32)
            # Use the MaskedAutoEncoder for sequence-to-sequence prediction
            self.args.ode = False
            
            # Fix dimensional mismatch by trimming the final position of o_pad and d_pad
            o_pad_fixed = o_pad[:, :-1]  # [b, o_seq_len] remove the last slot
            d_pad_fixed = d_pad[:, :-1]  # [b, d_seq_len] remove the last slot
            
            # Build a full representation of the source (hometown) sequence
            o_time_emb = self.time_embedding(o_t.to(torch.float32).unsqueeze(-1))  # [b, o_seq_len, d]
            o_space_emb = self.space_embedding(o_l.to(torch.float32))  # [b, o_seq_len, d]
            o_concat_emb = torch.cat([o_time_emb, o_space_emb, o_emb], dim=-1)  # [b, o_seq_len, 3*d]
            o_seq = self.concat_to_unified(o_concat_emb)  # [b, o_seq_len, d] unified projection
            
            # Build a full representation of the destination sequence
            d_time_emb = self.time_embedding(d_t.to(torch.float32).unsqueeze(-1))  # [b, d_seq_len, d]
            d_space_emb = self.space_embedding(d_l.to(torch.float32))  # [b, d_seq_len, d]
            d_concat_emb = torch.cat([d_time_emb, d_space_emb, d_target_emb], dim=-1)  # [b, d_seq_len, 3*d]
            d_seq = self.concat_to_unified(d_concat_emb)  # [b, d_seq_len, d] unified projection
            
            # ======================== Fully concatenate hometown and destination sequences ========================
            # Concatenate the hometown and destination sequences without inserting padding
            combined_seq_list = []
            combined_pad_list = []
            
            for i in range(batch_size):
                # Length of the valid portion in the hometown sequence
                o_valid_len = o_pad_fixed[i].sum().item()
                o_seq_valid = o_seq[i, :o_valid_len, :]  # [o_valid_len, d]
                
                # Length of the valid portion in the destination sequence
                d_valid_len = d_pad_fixed[i].sum().item()
                d_seq_valid = d_seq[i, :d_valid_len, :]  # [d_valid_len, d]
                
                # Concatenate without leaving gaps
                combined_seq = torch.cat([o_seq_valid, d_seq_valid], dim=0)  # [o_valid_len + d_valid_len, d]
                combined_len = combined_seq.shape[0]
                
                # Create a matching padding mask (True indicates a valid position)
                combined_pad = torch.ones(combined_len, dtype=torch.bool, device=self.args.device)
                
                combined_seq_list.append(combined_seq)
                combined_pad_list.append(combined_pad)
            
            # Pad sequences to the same length
            max_combined_len = max(seq.shape[0] for seq in combined_seq_list)
            combined_seq_padded = torch.zeros(batch_size, max_combined_len, self.hidden_size, device=self.args.device)
            combined_pad_padded = torch.zeros(batch_size, max_combined_len, dtype=torch.bool, device=self.args.device)
            
            # Track each sample's hometown length for extracting the destination segment later
            o_valid_lens = []
            d_valid_lens = []
            
            for i, (seq, pad) in enumerate(zip(combined_seq_list, combined_pad_list)):
                seq_len = seq.shape[0]
                combined_seq_padded[i, :seq_len, :] = seq
                combined_pad_padded[i, :seq_len] = pad
                
                # Record the original lengths
                o_valid_len = o_pad_fixed[i].sum().item()
                d_valid_len = d_pad_fixed[i].sum().item()
                o_valid_lens.append(o_valid_len)
                d_valid_lens.append(d_valid_len)
            
            # ======================== MaskedAutoEncoder processing ========================
            is_training = target_seq is not None
            
            # Determine the start/end positions of the destination sequence
            hometown_len_list = o_valid_lens  # Hometown sequence lengths
            destination_start_list = o_valid_lens  # Destination starts immediately after hometown
            destination_end_list = [o_valid_lens[i] + d_valid_lens[i] - 1 for i in range(batch_size)]  # Destination end indices
            
            if is_training:
                # Optionally use the LLM embedding as a cls_token
                if self.args.use_llm and generated_embeddings is not None:
                    self.mae.set_external_cls_token(generated_embeddings)
                else:
                    self.mae.set_external_cls_token(None)
                
                # During training: randomly mask parts of the destination while preserving endpoints
                loss, pred, mask = self.mae(
                    combined_seq_padded,
                    uid=uid,
                    mask_ratio=self.args.mask_ratio,  # Training-time mask ratio
                    hometown_len_list=hometown_len_list,
                    destination_start_list=destination_start_list,
                    destination_end_list=destination_end_list,
                    valid_mask=combined_pad_padded,
                    training=True
                )
                seq2seq_loss = loss
                
                # Extract destination predictions for downstream processing
                P_D_list = []
                for i in range(batch_size):
                    o_len = o_valid_lens[i]
                    d_len = d_valid_lens[i]
                    if d_len > 0:
                        d_pred = pred[i, o_len:o_len+d_len, :]  # Destination slice
                        # Pad back to the original destination length
                        d_pred_padded = torch.zeros(d_pad_fixed.shape[1], self.hidden_size, device=self.args.device)
                        d_pred_padded[:d_len, :] = d_pred
                        P_D_list.append(d_pred_padded)
                    else:
                        # If no valid destination sequence exists, create zero padding
                        d_pred_padded = torch.zeros(d_pad_fixed.shape[1], self.hidden_size, device=self.args.device)
                        P_D_list.append(d_pred_padded)
                
                P_D = torch.stack(P_D_list, dim=0)  # [b, d_seq_len, d]
                
            else:
                # Optionally set the LLM embedding as cls_token
                if self.args.use_llm and generated_embeddings is not None:
                    self.mae.set_external_cls_token(generated_embeddings)
                else:
                    self.mae.set_external_cls_token(None)
                
                # Inference: mask destination tokens except for start and end points
                pred, mask = self.mae(
                    combined_seq_padded,
                    uid=uid,
                    mask_ratio=self.args.mask_ratio,  # Not applied during inference
                    hometown_len_list=hometown_len_list,
                    destination_start_list=destination_start_list,
                    destination_end_list=destination_end_list,
                    valid_mask=combined_pad_padded,
                    training=False
                )
                
                # Extract destination predictions
                P_D_list = []
                for i in range(batch_size):
                    o_len = o_valid_lens[i]
                    d_len = d_valid_lens[i]
                    if d_len > 0:
                        d_pred = pred[i, o_len:o_len+d_len, :]  # Destination slice
                        # Pad back to the original destination length
                        d_pred_padded = torch.zeros(d_pad_fixed.shape[1], self.hidden_size, device=self.args.device)
                        d_pred_padded[:d_len, :] = d_pred
                        P_D_list.append(d_pred_padded)
                    else:
                        # If no valid destination sequence exists, create zero padding
                        d_pred_padded = torch.zeros(d_pad_fixed.shape[1], self.hidden_size, device=self.args.device)
                        P_D_list.append(d_pred_padded)
                
                P_D = torch.stack(P_D_list, dim=0)  # [b, d_seq_len, d]
            
            # Apply destination padding mask to suppress invalid positions
            P_D = P_D * d_pad_fixed.unsqueeze(-1)  # Zero out invalid slots

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
        poi_output = self.predictor(encoder_output)
        masked_poi_output = poi_output.masked_fill(~region_mask_uns, -1e9)
        if target_seq is not None:
            loss = self.criterion(masked_poi_output.view(-1, self.poi_size), d_ck.flatten())
            if self.args.st_module:
                loss += seq2seq_loss
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
    
    def set_mae_llm_embedding(self, P_L):
        """
        Provide the MaskedAutoEncoder with LLM embeddings as semantic guidance.
        Args:
            P_L: [N, D] LLM embeddings used to derive semantic-aware masking strategies.
        """
        if hasattr(self, 'mae') and self.mae is not None:
            self.mae.set_external_cls_token(P_L)
        else:
            print("Warning: MaskedAutoEncoder is not initialized. Please enable st_module in args.")