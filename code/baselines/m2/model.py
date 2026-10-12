"""
Mask DINO with Query Selection v2 for protein domain segmentation + function prediction.

Strictly aligned with hemenet/query_v2/mask_dino_model.py architecture.

Components:
1. Sinusoidal positional encoding injected into mask branch (pixel embedding)
2. Denoising Training: GT spans + noise as extra decoder queries with direct supervision
3. Domain-aware Query Selection: scoring head outputs used for auxiliary domain loss
4. Attention mask to isolate DN queries from normal queries in decoder
"""

import torch
import torch.nn as nn
import math


def sinusoidal_position_encoding(length, dim, device=None):
    """Generate sinusoidal positional encoding [length, dim]."""
    pe = torch.zeros(length, dim, device=device)
    position = torch.arange(0, length, device=device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * -(math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [length, dim]


class QuerySelectionMaskDinoV2(nn.Module):
    def __init__(
        self,
        hidden_dim=1280,
        num_queries=8,
        num_dn_queries=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        nhead=8,
        dim_feedforward=2048,
        num_classes=256,
        task_type="classification",
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_dn_queries = num_dn_queries
        self.task_type = task_type
        self.hidden_dim = hidden_dim

        # ============ Query Selection ============
        # Scoring head: scores each encoder position for query selection
        # Also used for domain-aware auxiliary loss
        self.query_score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Anchor MLP: maps normalized position [0,1] to position embedding
        self.anchor_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

        # ============ Denoising Query Construction ============
        # DN anchor MLP: maps noisy normalized positions to embeddings
        self.dn_anchor_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),  # input: [center, width] normalized
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )
        # DN content projection: project memory features for DN queries
        self.dn_content_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ============ Encoder ============
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        # ============ Decoder ============
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True, dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        # ============ Mask Head with Positional Encoding ============
        self.mask_embed = nn.Linear(hidden_dim, hidden_dim)
        self.mask_gate = nn.Linear(hidden_dim, hidden_dim)
        # Learnable scale for positional encoding injection
        self.pe_scale = nn.Parameter(torch.ones(1) * 0.1)

        # ============ Task Heads ============
        if task_type == "regression":
            self.regression_head = nn.Linear(hidden_dim, 1)
        else:
            self.class_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, residue_feat, residue_mask, gt_spans=None, training=False):
        """
        Args:
            residue_feat: [B, L, D] from ESM backbone (pre-computed)
            residue_mask: [B, L] attention mask (1=valid, 0=pad)
            gt_spans: List[List[Tuple[int, int]]] per-sample GT domain spans
                      Only used during training for denoising queries.
            training: bool, whether in training mode

        Returns:
            dict with keys:
                - mask_logits: [B, Q, L] (normal queries only)
                - class_logits: [B, Q, C] from normal queries
                - query_scores: [B, L] raw scores from scoring head (for aux loss)
                - dn_mask_logits: [B, K, L] (denoising queries, only during training)
                - dn_gt_masks: List[List[Tensor]] (GT masks for DN queries)
        """
        B, L, D = residue_feat.shape
        Q = self.num_queries
        pad_mask = (residue_mask == 0)  # True=pad for PyTorch

        # ============ Encode ============
        memory = self.encoder(residue_feat, src_key_padding_mask=pad_mask)

        # ============ Query Selection ============
        scores = self.query_score_head(memory).squeeze(-1)  # [B, L]
        scores_masked = scores.masked_fill(pad_mask, -1e4)

        # Select top-k positions
        _, topk_indices = scores_masked.topk(Q, dim=1)  # [B, Q]
        topk_indices, _ = topk_indices.sort(dim=1)  # [B, Q]

        # Init Contents: gather encoder features at top-k positions
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, Q, D]
        content_queries = torch.gather(memory, 1, topk_indices_expanded)  # [B, Q, D]

        # Init Anchors: position-aware embedding from normalized positions
        seq_lens = residue_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
        normalized_positions = topk_indices.float() / seq_lens.clamp(min=1)  # [B, Q]
        anchor_embed = self.anchor_mlp(normalized_positions.unsqueeze(-1))  # [B, Q, D]

        # Combine: content + anchor
        normal_queries = content_queries + anchor_embed  # [B, Q, D]

        # ============ Denoising Queries (training only) ============
        dn_mask_logits = None
        dn_gt_masks = None

        if training and gt_spans is not None:
            dn_queries, dn_gt_masks = self._build_dn_queries(
                memory, residue_mask, gt_spans, seq_lens
            )  # dn_queries: [B, K, D]
            K = dn_queries.size(1)

            # Concatenate normal + DN queries
            all_queries = torch.cat([normal_queries, dn_queries], dim=1)  # [B, Q+K, D]

            # Build attention mask to prevent DN queries from attending to normal queries
            total_q = Q + K
            tgt_mask = torch.zeros(total_q, total_q, device=residue_feat.device, dtype=torch.bool)
            tgt_mask[Q:, :Q] = True
            tgt_mask[:Q, Q:] = True

            # Decode all queries together
            h_all = self.decoder(
                all_queries, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=pad_mask,
            )  # [B, Q+K, D]

            h_q = h_all[:, :Q, :]  # [B, Q, D] normal
            h_dn = h_all[:, Q:, :]  # [B, K, D] denoising
        else:
            # Inference: only normal queries
            h_q = self.decoder(
                normal_queries, memory,
                memory_key_padding_mask=pad_mask,
            )  # [B, Q, D]

        # ============ Mask prediction with positional encoding ============
        pos_enc = sinusoidal_position_encoding(L, D, device=residue_feat.device)  # [L, D]
        pos_enc = pos_enc.unsqueeze(0)  # [1, L, D]

        gate = torch.sigmoid(self.mask_gate(memory))  # [B, L, D]
        mask_feat = gate * (residue_feat + self.pe_scale * pos_enc)  # [B, L, D]

        # Normal query mask logits
        mask_logits = torch.einsum(
            "bqd,bld->bql",
            self.mask_embed(h_q), mask_feat,
        )  # [B, Q, L]

        # DN query mask logits (training only)
        if training and gt_spans is not None:
            dn_mask_logits = torch.einsum(
                "bkd,bld->bkl",
                self.mask_embed(h_dn), mask_feat,
            )  # [B, K, L]

        # ============ Task-specific head ============
        result = {
            "mask_logits": mask_logits,
            "query_scores": scores,  # [B, L] for domain-aware aux loss
        }

        if self.task_type == "regression":
            result["regression_output"] = self.regression_head(h_q)  # [B, Q, 1]
        else:
            result["class_logits"] = self.class_head(h_q)  # [B, Q, C]

        if training and dn_mask_logits is not None:
            result["dn_mask_logits"] = dn_mask_logits
            result["dn_gt_masks"] = dn_gt_masks

        return result

    def _build_dn_queries(self, memory, residue_mask, gt_spans, seq_lens):
        """
        Build denoising queries from GT spans with noise.

        Args:
            memory: [B, L, D] encoder output
            residue_mask: [B, L]
            gt_spans: List[List[Tuple[int, int]]] per-sample GT spans
            seq_lens: [B, 1] valid sequence lengths

        Returns:
            dn_queries: [B, K, D] denoising queries
            dn_gt_masks: List[List[Tensor[L]]] per-sample GT masks for each DN query
        """
        B, L, D = memory.shape
        K = self.num_dn_queries
        device = memory.device

        dn_queries_list = []
        dn_gt_masks_list = []

        for b in range(B):
            spans = gt_spans[b]  # List[(start, end)]
            seq_len = int(seq_lens[b].item())

            if len(spans) == 0:
                # No GT spans: use zero queries
                dn_q = torch.zeros(K, D, device=device)
                dn_gt = [torch.zeros(L, device=device) for _ in range(K)]
            else:
                dn_q_list = []
                dn_gt_list = []

                for k in range(K):
                    # Randomly pick a GT span
                    span_idx = torch.randint(0, len(spans), (1,)).item()
                    start, end = spans[span_idx]

                    # Add noise to center and width
                    center = (start + end) / 2.0
                    width = end - start

                    # Position noise: ±10% of sequence length
                    center_noise = (torch.rand(1, device=device).item() - 0.5) * 0.2 * seq_len
                    # Scale noise: 0.7x ~ 1.3x
                    scale_noise = 0.7 + torch.rand(1, device=device).item() * 0.6

                    noisy_center = center + center_noise
                    noisy_width = width * scale_noise

                    # Clamp to valid range
                    noisy_center = max(0, min(seq_len - 1, noisy_center))
                    noisy_width = max(5, min(seq_len, noisy_width))

                    # Normalized features for anchor MLP
                    norm_center = noisy_center / max(seq_len, 1)
                    norm_width = noisy_width / max(seq_len, 1)
                    anchor_input = torch.tensor(
                        [norm_center, norm_width], device=device, dtype=torch.float32
                    )
                    anchor_emb = self.dn_anchor_mlp(anchor_input.unsqueeze(0))  # [1, D]

                    # Content: average pool memory around noisy center
                    noisy_start = int(max(0, noisy_center - noisy_width / 2))
                    noisy_end = int(min(seq_len, noisy_center + noisy_width / 2))
                    if noisy_end <= noisy_start:
                        noisy_end = noisy_start + 1
                    content = memory[b, noisy_start:noisy_end, :].mean(dim=0, keepdim=True)  # [1, D]
                    content = self.dn_content_proj(content)  # [1, D]

                    dn_q_list.append(content + anchor_emb)  # [1, D]

                    # GT mask for this DN query (the original span, not noisy)
                    gt_mask = torch.zeros(L, device=device)
                    gt_mask[start:end] = 1.0
                    dn_gt_list.append(gt_mask)

                dn_q = torch.cat(dn_q_list, dim=0)  # [K, D]
                dn_gt = dn_gt_list

            dn_queries_list.append(dn_q)
            dn_gt_masks_list.append(dn_gt)

        dn_queries = torch.stack(dn_queries_list, dim=0)  # [B, K, D]
        return dn_queries, dn_gt_masks_list
