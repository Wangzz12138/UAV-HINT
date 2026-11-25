"""Paper Sec. 4: hierarchical queries and progressive frozen-T5 decoding.

The paper does not specify a cross-attention FFN, gate predictor or pooling
operator. This implementation uses one shared cross-attention layer, learned
sigmoid token gates, a two-layer GELU projector, and per-level mean pooling.
"""
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack


LEVELS = ("action", "short_term", "long_term")


@dataclass
class ModelConfig:
    feature_dim: int = 512
    num_queries: int = 16
    num_prefix: int = 16
    cross_attention_heads: int = 8
    dropout: float = 0.1
    num_classes: tuple = (7, 8, 4)
    t5_model: str = "google-t5/t5-base"
    t5_revision: Optional[str] = None
    local_files_only: bool = False
    use_hqr: bool = True
    use_ptu: bool = True
    # These fields are ONLY for explicitly requested random-decoder smoke runs.
    random_decoder: bool = False
    decoder_dim: int = 768
    decoder_layers: int = 12
    decoder_heads: int = 12
    decoder_ff_dim: int = 3072

    def to_dict(self):
        return asdict(self)


class HierarchicalQueryRefinement(nn.Module):
    """Eq. (3–4): one scalar gate per token, broadcast over channels."""

    def __init__(self, num_queries=16, feature_dim=512, enabled=True):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(3, num_queries, feature_dim))
        self.short_gate = nn.Parameter(torch.zeros(num_queries, 1))
        self.long_gate = nn.Parameter(torch.zeros(num_queries, 1))
        self.enabled = enabled
        if not enabled:
            self.short_gate.requires_grad_(False)
            self.long_gate.requires_grad_(False)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self):
        action, short, long = self.queries.unbind(0)
        if self.enabled:
            m_s, m_l = self.short_gate.sigmoid(), self.long_gate.sigmoid()
            short = m_s * short + (1 - m_s) * action
            long = m_l * long + (1 - m_l) * short
        return torch.stack((action, short, long))


def progressive_attention_mask(layer_index, num_layers, num_prefix, num_queries,
                               *, device=None, dtype=torch.float32, enabled=True):
    """Algorithm 1, zero-based layer index; mask KEY columns for every query.

    No causal triangle is applied. All tokens keep residual states throughout;
    'unveiling' controls key visibility, not token removal or query-row masking.
    """
    if not 0 <= layer_index < num_layers:
        raise ValueError("layer_index must be in [0, num_layers)")
    if num_layers < 3 or num_prefix < 0 or num_queries < 1:
        raise ValueError("PTU needs >=3 layers and positive query count")
    length = num_prefix + 3 * num_queries
    phase = 0 if 3 * (layer_index + 1) <= num_layers else (
        1 if 3 * (layer_index + 1) <= 2 * num_layers else 2)
    visible = num_prefix + (phase + 1) * num_queries if enabled else length
    mask = torch.zeros(1, 1, length, length, device=device, dtype=dtype)
    mask[..., visible:] = float("-inf")
    return mask


class ProgressiveT5Decoder(nn.Module):
    """Frozen T5 blocks with per-layer PTU masks and preserved input gradients."""

    def __init__(self, config: ModelConfig, decoder_config=None):
        super().__init__()
        if decoder_config is not None:
            # Used when restoring saved decoder weights without a second download.
            self.stack = T5Stack(T5Config.from_dict(decoder_config))
        elif config.random_decoder:
            t5_config = T5Config(
                vocab_size=32, d_model=config.decoder_dim,
                d_ff=config.decoder_ff_dim, num_layers=config.decoder_layers,
                num_decoder_layers=config.decoder_layers,
                num_heads=config.decoder_heads,
                d_kv=config.decoder_dim // config.decoder_heads,
                dropout_rate=config.dropout, is_decoder=True,
                is_encoder_decoder=False, use_cache=False,
            )
            self.stack = T5Stack(t5_config)
        else:
            pretrained = T5ForConditionalGeneration.from_pretrained(
                config.t5_model, revision=config.t5_revision,
                local_files_only=config.local_files_only)
            self.stack = pretrained.decoder
            # No discrete token IDs are used; avoid retaining unused embeddings.
            self.stack.embed_tokens = None
        if len(self.stack.block) < 3:
            raise ValueError("The decoder must have at least three layers for PTU")
        self.stack.requires_grad_(False)
        self.stack.eval()
        self.num_prefix = config.num_prefix
        self.num_queries = config.num_queries
        self.use_ptu = config.use_ptu
        self.hidden_size = self.stack.config.d_model

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, inputs_embeds, return_attentions=False):
        # Do NOT use torch.no_grad(): frozen parameters still transmit gradients
        # to learnable queries, projection and prefix through all decoder blocks.
        hidden = self.stack.dropout(inputs_embeds)
        length = hidden.shape[1]
        expected = self.num_prefix + 3 * self.num_queries
        if length != expected:
            raise ValueError(f"Expected {expected} decoder tokens, got {length}")
        first_attention = self.stack.block[0].layer[0].SelfAttention
        base_bias = first_attention.compute_bias(length, length, device=hidden.device)
        base_bias = base_bias.to(hidden.dtype)
        attentions = []
        for index, block in enumerate(self.stack.block):
            mask = progressive_attention_mask(
                index, len(self.stack.block), self.num_prefix, self.num_queries,
                device=hidden.device, dtype=hidden.dtype, enabled=self.use_ptu)
            # T5 normally reuses a bias already containing the first mask. Build
            # fresh bias each layer so later phases really unveil new columns.
            outputs = block(
                hidden, attention_mask=None, position_bias=base_bias + mask,
                encoder_hidden_states=None, use_cache=False,
                output_attentions=return_attentions,
            )
            hidden = outputs[0]
            if return_attentions:
                attentions.append(outputs[-1])
        hidden = self.stack.dropout(self.stack.final_layer_norm(hidden))
        return (hidden, attentions) if return_attentions else hidden


class HINTNet(nn.Module):
    """Full HINT-Net, accepting raw modalities or cached ActionCLIP tokens."""

    def __init__(self, config: Optional[ModelConfig] = None, backbone=None, decoder_config=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        if cfg.num_queries < 1 or cfg.num_prefix < 1:
            raise ValueError("num_queries and num_prefix must be positive")
        if len(cfg.num_classes) != 3 or any(c < 2 for c in cfg.num_classes):
            raise ValueError("num_classes must describe three classification levels")
        self.backbone = backbone
        self.hqr = HierarchicalQueryRefinement(
            cfg.num_queries, cfg.feature_dim, cfg.use_hqr)
        self.cross_attention = nn.MultiheadAttention(
            cfg.feature_dim, cfg.cross_attention_heads,
            dropout=cfg.dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(cfg.feature_dim)
        self.decoder = ProgressiveT5Decoder(cfg, decoder_config=decoder_config)
        decoder_dim = self.decoder.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(cfg.feature_dim, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim))
        self.prefix = nn.Parameter(torch.empty(1, cfg.num_prefix, decoder_dim))
        nn.init.normal_(self.prefix, std=0.02)
        self.heads = nn.ModuleDict({
            name: nn.Linear(decoder_dim, classes)
            for name, classes in zip(LEVELS, cfg.num_classes)
        })

    def forward(self, tokens=None, token_mask=None, video=None, local_video=None,
                text=None, return_features=False):
        if tokens is None:
            if self.backbone is None:
                raise ValueError("Supply cached tokens or configure an ActionCLIP backbone")
            tokens, token_mask = self.backbone(video, local_video, text)
        if tokens.ndim != 3 or tokens.shape[-1] != self.config.feature_dim:
            raise ValueError(f"tokens must be [B,S,{self.config.feature_dim}]")
        if tokens.shape[1] == 0 or not torch.isfinite(tokens).all():
            raise ValueError("tokens must be nonempty and finite")
        tokens = tokens.to(dtype=self.prefix.dtype)
        if token_mask is None:
            token_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if token_mask.shape != tokens.shape[:2]:
            raise ValueError("token_mask must be [B,S]")
        token_mask = token_mask.bool()
        if not token_mask.any(dim=1).all():
            raise ValueError("Every sample must have at least one visible input token")
        queries = self.hqr().flatten(0, 1)
        queries = queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        features, _ = self.cross_attention(
            queries, tokens, tokens, key_padding_mask=~token_mask, need_weights=False)
        features = self.query_norm(features + queries)
        projected = self.projection(features)
        decoder_input = torch.cat((self.prefix.expand(tokens.shape[0], -1, -1), projected), dim=1)
        hidden = self.decoder(decoder_input)
        hierarchy = hidden[:, self.config.num_prefix:].reshape(
            tokens.shape[0], 3, self.config.num_queries, -1).mean(dim=2)
        logits = {name: self.heads[name](hierarchy[:, index])
                  for index, name in enumerate(LEVELS)}
        if return_features:
            return logits, {"hierarchical_features": hierarchy, "decoder_states": hidden}
        return logits
