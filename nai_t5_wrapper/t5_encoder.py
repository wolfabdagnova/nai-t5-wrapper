import math
from typing import Optional, Callable, Protocol, TYPE_CHECKING, NamedTuple, Any
from itertools import chain
from functools import partial

import torch
from einops import rearrange
from torch import BoolTensor, FloatTensor, IntTensor, LongTensor, nn, __version__ as torch_version
from torch.nn import Linear, Embedding
from torch.nn.functional import scaled_dot_product_attention

from .t5_common import (
    RMSNormCast,
    T5AttnImpl,
    T5GEGLUFFN,
    T5Config,
    T5RelativeAttentionBias,
    T5ReLUFFN,
    ActAndResidual,
    get_ffn_factory,
    init_emb,
)
from .flex_utils import ScoreMod, MaskMod

if TYPE_CHECKING:
    # avoid runtime import to prevent explosion on older PyTorch
    from torch.nn.attention.flex_attention import BlockMask
else:
    BlockMask = _mask_mod_signature = Any

try:
    from torch.nn.attention.flex_attention import create_block_mask
    if torch_version >= '2.6.0a0':
        create_block_mask_c = torch.compile(create_block_mask, dynamic=False, fullgraph=True)
    else:
        create_block_mask_c = partial(create_block_mask, _compile=True)
except ImportError:
    create_block_mask_c = None

####
#### Attention argument types
####


class SDPAArgs(NamedTuple):
    position_bias: FloatTensor
    mask: Optional[BoolTensor]


class FlexArgs(NamedTuple):
    block_mask: Optional[BlockMask]


####
#### T5 encoder self-attention (SDPA)
####


class T5EncoderSelfAttention(nn.Module):
    qkv_proj: Linear
    o_proj: Linear
    head_dim: int
    scale: float
    dropout: float
    config: T5Config

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        assert config.n_head == config.kv_heads, "Q and KV heads must be equal; GQA not implemented yet."
        self.head_dim = config.head_dim
        qkv_heads: int = config.n_head + config.kv_heads * 2
        self.scale = self.head_dim**-0.5 if config.scale_qk else 1.0
        self.dropout = config.dropout
        self.qkv_proj = Linear(
            in_features=config.hidden_dim,
            out_features=config.head_dim * qkv_heads,
            bias=False,
            dtype=config.linear_weight_dtype,
        )
        self.o_proj = Linear(
            in_features=config.head_dim * config.n_head,
            out_features=config.hidden_dim,
            bias=False,
            dtype=config.linear_weight_dtype,
        )
        self.config = config

    def forward(
        self,
        x: FloatTensor,
        position_bias: FloatTensor,
        mask: Optional[BoolTensor] = None,
    ) -> FloatTensor:
        qkv: FloatTensor = self.qkv_proj(x)
        q, k, v = rearrange(
            qkv, "batch seq (proj heads head_dim) -> proj batch heads seq head_dim", proj=3, head_dim=self.head_dim
        ).unbind()

        attn_mask = position_bias.type_as(q)
        if mask is not None:
            attn_mask = attn_mask.masked_fill(~mask, torch.finfo(q.dtype).min)

        a = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        a = rearrange(a, "batch heads seq head_dim -> batch seq (heads head_dim)")
        o = self.o_proj(a)
        return o

    def init_weights(self, generator: Optional[torch.Generator] = None) -> None:
        nn.init.normal_(self.qkv_proj.weight, std=1 / math.sqrt(self.config.hidden_dim), generator=generator)
        nn.init.normal_(self.o_proj.weight, std=1 / math.sqrt(self.config.hidden_dim * self.config.num_layers), generator=generator)


####
#### T5 encoder self-attention (Flex)
####


class T5EncoderSelfAttentionFlex(nn.Module):
    qkv_proj: Linear
    o_proj: Linear
    head_dim: int
    scale: float
    dropout: float
    config: T5Config
    score_mod: Optional[ScoreMod]

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        assert config.n_head == config.kv_heads, "Q and KV heads must be equal; GQA not implemented yet."
        self.head_dim = config.head_dim
        qkv_heads: int = config.n_head + config.kv_heads * 2
        self.scale = self.head_dim**-0.5 if config.scale_qk else 1.0
        self.dropout = config.dropout
        self.qkv_proj = Linear(
            in_features=config.hidden_dim,
            out_features=config.head_dim * qkv_heads,
            bias=False,
            dtype=config.linear_weight_dtype,
        )
        self.o_proj = Linear(
            in_features=config.head_dim * config.n_head,
            out_features=config.hidden_dim,
            bias=False,
            dtype=config.linear_weight_dtype,
        )
        self.config = config
        self.score_mod = None

    def make_score_mod(self, bias: FloatTensor) -> ScoreMod:
        # bias shape: (heads, q_len, k_len) or (1, heads, q_len, k_len)
        if bias.ndim == 4:
            bias = bias.squeeze(0)

        def score_mod(
            score: FloatTensor,
            batch: IntTensor,
            head: IntTensor,
            q_idx: IntTensor,
            kv_idx: IntTensor,
        ) -> FloatTensor:
            return score + bias[head, q_idx, kv_idx]

        return score_mod

    def forward(
        self,
        x: FloatTensor,
        block_mask: Optional[BlockMask] = None,
    ) -> FloatTensor:
        from torch.nn.attention.flex_attention import flex_attention

        qkv: FloatTensor = self.qkv_proj(x)
        q, k, v = rearrange(
            qkv, "batch seq (proj heads head_dim) -> proj batch heads seq head_dim", proj=3, head_dim=self.head_dim
        ).unbind()

        a = flex_attention(
            q,
            k,
            v,
            score_mod=self.score_mod,
            block_mask=block_mask,
            scale=self.scale,
        )
        a = rearrange(a, "batch heads seq head_dim -> batch seq (heads head_dim)")
        o = self.o_proj(a)
        return o

    def init_weights(self, generator: Optional[torch.Generator] = None) -> None:
        nn.init.normal_(self.qkv_proj.weight, std=1 / math.sqrt(self.config.hidden_dim), generator=generator)
        nn.init.normal_(self.o_proj.weight, std=1 / math.sqrt(self.config.hidden_dim * self.config.num_layers), generator=generator)


####
#### T5 encoder layer
####


class T5EncoderLayer(nn.Module):
    attn: T5EncoderSelfAttention | T5EncoderSelfAttentionFlex
    ln1: RMSNormCast
    """pre-attn layer norm"""
    ln2: RMSNormCast
    """pre-ffn layer norm"""
    ffn: T5ReLUFFN | T5GEGLUFFN
    dropout: nn.Dropout

    def __init__(self, config: T5Config, attn_ctor: type) -> None:
        super().__init__()
        self.attn = attn_ctor(config)
        self.ln1 = RMSNormCast(config.hidden_dim, eps=config.eps, dtype=config.norm_weight_dtype, elementwise_affine=config.elementwise_affine)
        self.ln2 = RMSNormCast(config.hidden_dim, eps=config.eps, dtype=config.norm_weight_dtype, elementwise_affine=config.elementwise_affine)
        ffn_factory = get_ffn_factory(config.ffn_type)
        self.ffn = ffn_factory(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x_r: ActAndResidual,
        attn_args: SDPAArgs | FlexArgs,
    ) -> ActAndResidual:
        x, residual = x_r
        x, residual = self.ln1(x, residual=residual)
        x = self.attn(x, *attn_args)
        x, residual = self.ln2(self.dropout(x), residual=residual)
        x = self.ffn(x)
        return ActAndResidual(self.dropout(x), residual)

    def init_weights(self, generator: Optional[torch.Generator] = None) -> None:
        self.attn.init_weights(generator)
        self.ln1.reset_parameters()
        self.ln2.reset_parameters()
        self.ffn.init_weights(generator)


####
#### Block mask helper for Flex Attention
####


def make_self_attn_block_mask(
    mask: BoolTensor,
    mask_pad_queries: bool = True,
    create_block_mask: Optional[Callable] = None,
) -> BlockMask:
    """Create a block mask for encoder self-attention with padding support.

    Args:
        mask: Bool padding mask of shape (batch, seq_len). True = real token, False = padding.
        mask_pad_queries: If True, padding queries also don't attend to anything.
        create_block_mask: The create_block_mask function to use. If None, uses the
            module-level compiled version or imports the default.
    """
    B, S = mask.shape

    def mask_mod(
        batch: IntTensor,
        head: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        kv_valid = mask[batch, kv_idx]
        if mask_pad_queries:
            q_valid = mask[batch, q_idx]
            return kv_valid & q_valid
        return kv_valid

    if create_block_mask is None:
        if create_block_mask_c is not None:
            create_block_mask = create_block_mask_c
        else:
            from torch.nn.attention.flex_attention import create_block_mask as _cbm
            create_block_mask = _cbm

    return create_block_mask(mask_mod, B, 1, S, S, device=mask.device)


####
#### T5 encoder stack
####


class T5EncoderStack(nn.Module):
    config: T5Config
    vocab_embed: Embedding
    relative_attention_bias: T5RelativeAttentionBias
    dropout: nn.Dropout
    layers: nn.ModuleList
    ln: RMSNormCast
    param_count: int
    non_emb_param_count: int

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.config = config
        self.vocab_embed = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_dim,
            dtype=config.emb_weight_dtype,
        )
        self.relative_attention_bias = T5RelativeAttentionBias(config, bidirectional=True)
        match config.attn_impl:
            case T5AttnImpl.SDPA:
                attn_ctor = T5EncoderSelfAttention
            case T5AttnImpl.Flex:
                attn_ctor = T5EncoderSelfAttentionFlex
            case _:
                raise ValueError(f"Unsupported attention implementation: {config.attn_impl}")
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([T5EncoderLayer(config, attn_ctor) for _ in range(config.num_layers)])
        self.ln = RMSNormCast(config.hidden_dim, eps=config.eps, dtype=config.norm_weight_dtype, elementwise_affine=config.elementwise_affine)
        # we must calculate this at init, and not later, because FSDP may shard the params
        self.param_count = emb_param_count = 0
        for p in self.parameters():
            self.param_count += p.numel()
        for p in chain(self.vocab_embed.parameters(), self.relative_attention_bias.parameters()):
            emb_param_count += p.numel()
        self.non_emb_param_count = self.param_count - emb_param_count

    def forward(
        self,
        input_ids: LongTensor,
        input_mask: Optional[BoolTensor] = None,
        # SDPA: position_bias can be re-used across invocations; best computed outside
        position_bias: Optional[FloatTensor] = None,
        # Flex: we externalize BlockMask so that you can manage how create_block_mask is compiled (including it inside the model would require torch 2.6)
        block_mask: Optional[BlockMask] = None,
    ) -> FloatTensor:
        seq_len: int = input_ids.size(-1)
        input_embeds = self.vocab_embed(input_ids.flatten(end_dim=-2))

        attn_args: list[SDPAArgs|FlexArgs]
        match self.config.attn_impl:
            case T5AttnImpl.SDPA:
                if position_bias is None:
                    seq_len: int = input_ids.size(-1)
                    position_bias: FloatTensor = self.relative_attention_bias(seq_len)

                # UMT5 has a position embedding per layer, and we computed all of them simultaneously
                # note: [t]*n does not duplicate tensors, it just makes a list of n references to the same tensor
                biases: list[FloatTensor] = position_bias.unbind() if self.config.pos_emb_per_layer else [position_bias]*self.config.num_layers

                attn_mask: Optional[BoolTensor] = None if input_mask is None else self.broadcast_mask(input_mask, input_ids)

                attn_args = [SDPAArgs(
                    position_bias=bias,
                    mask=attn_mask,
                ) for bias in biases]
            case T5AttnImpl.Flex:
                attn_args = [FlexArgs(block_mask=block_mask)]*self.config.num_layers

        x_r = ActAndResidual(self.dropout(input_embeds), None)
        for layer, attn_args_ in zip(self.layers, attn_args):
            layer: T5EncoderLayer
            x_r = layer(x_r, attn_args_)
        x, residual = x_r
        x = self.ln(x, residual=residual, prenorm=False)
        return x

    def bind_score_mods(self, seq_len=512) -> None:
        """
        [Only relevant if you're using flex attention]
        Run this function after you have loaded weights and transferred the model to device.
        You will probably want to run this within an inference_mode() or no_grad() context!
        """
        assert self.config.attn_impl == T5AttnImpl.Flex, "Only Flex attention has score_mods"
        position_bias: FloatTensor = self.relative_attention_bias(seq_len)
        biases: list[FloatTensor] = position_bias.unbind() if self.config.pos_emb_per_layer else [position_bias.squeeze(0)]*self.config.num_layers
        for layer, bias in zip(self.layers, biases):
            layer: T5EncoderLayer
            attn: T5EncoderSelfAttentionFlex = layer.attn
            attn.score_mod = attn.make_score_mod(bias)

    def broadcast_mask(self, input_mask: BoolTensor, input_ids: LongTensor) -> BoolTensor:
        seq_len: int = input_ids.size(-1)
        scores_shape = torch.Size((input_ids.size(0), self.config.n_head, seq_len, seq_len))
        match input_mask.ndim:
            case 2:
                assert (
                    input_mask.shape == input_ids.shape
                ), f"attn mask was 2-dim, so expected padding mask: (batch, seq_len). got {input_mask.shape} mask for {input_ids.shape} inputs."
                # broadcast the key-mask over all heads and queries (b h q k)
                attn_mask = rearrange(input_mask, "b k -> b 1 1 k")
            case 3:
                assert input_mask.shape == torch.Size(
                    [*input_ids.shape, input_ids.shape[-1]]
                ), f"attn mask was 3-dim, so expected packing mask: (batch, seq_len, seq_len). got {input_mask.shape} mask for {input_ids.shape} inputs."
                # broadcast over all heads (b h q k)
                attn_mask = rearrange(input_mask, "b q k -> b 1 q k")
            case 4:
                try:
                    torch.broadcast_shapes(input_mask.shape, scores_shape)
                except RuntimeError:
                    raise ValueError(
                        f"input_mask ({input_mask.shape}) was not broadcastable to scores, (batch, heads, seq_len, seq_len) ({scores_shape})"
                    )
            case _:
                raise ValueError(
                    f"Expected to broadcast 2~4-dim input mask onto {scores_shape} scores, got mask {input_mask.shape} for {input_ids.shape} inputs."
                )
        return attn_mask

    def init_weights(self, generator: Optional[torch.Generator] = None) -> None:
        init_emb(self.vocab_embed, generator=generator)
        self.relative_attention_bias.init_weights(generator)
        self.ln.reset_parameters()
        for layer in self.layers:
            layer: T5EncoderLayer
            layer.init_weights(generator)
