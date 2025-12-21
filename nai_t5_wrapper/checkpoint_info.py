from enum import Enum
from typing import Optional

class Checkpoint(str, Enum):
    # T5 v1.1 variants
    T5v1_1Small = 't5-v1.1-small'
    T5v1_1XL = 't5-v1.1-xl'
    T5v1_1XXL = 't5-v1.1-xxl'
    T5v1Large = 't5-v1-large'
    # MT5 variants (Multilingual T5)
    MT5Small = 'mt5-small'
    MT5Base = 'mt5-base'
    MT5Large = 'mt5-large'
    MT5XL = 'mt5-xl'
    MT5XXL = 'mt5-xxl'
    # UMT5 variants (Unified Multilingual T5 - per-layer position embeddings)
    UMT5Small = 'umt5-small'
    UMT5Base = 'umt5-base'
    UMT5XL = 'umt5-xl'
    UMT5XXL = 'umt5-xxl'
    # Other T5 variants
    PileT5Large = 'pile-t5-large'

enc_ffn_out_scale_dict: dict[Checkpoint, Optional[list[float]]] = {
    # T5 v1.1 - 8 layers
    Checkpoint.T5v1_1Small: [*[1]*2, 1/2, *[1]*4, 1/2],
    # T5 v1.1 - 24 layers
    Checkpoint.T5v1_1XL: [1, 1/2, *[1]*3, 1/4, *[1]*18],
    # T5 v1.1 - 24 layers
    Checkpoint.T5v1_1XXL: [*[1]*7, 1/4, *[1]*16],
    # MT5 - scales not profiled yet, use None (no scaling)
    Checkpoint.MT5Small: None,
    Checkpoint.MT5Base: None,
    Checkpoint.MT5Large: None,
    Checkpoint.MT5XL: None,
    Checkpoint.MT5XXL: None,
    # UMT5 - scales not profiled yet, use None (no scaling)
    Checkpoint.UMT5Small: None,
    Checkpoint.UMT5Base: None,
    Checkpoint.UMT5XL: None,
    Checkpoint.UMT5XXL: None,
}

enc_attn_out_scale_dict: dict[Checkpoint, Optional[list[float]]] = {
    Checkpoint.T5v1_1Small: None,
    Checkpoint.T5v1_1XL: None,
    Checkpoint.T5v1_1XXL: None,
    Checkpoint.MT5Small: None,
    Checkpoint.MT5Base: None,
    Checkpoint.MT5Large: None,
    Checkpoint.MT5XL: None,
    Checkpoint.MT5XXL: None,
    Checkpoint.UMT5Small: None,
    Checkpoint.UMT5Base: None,
    Checkpoint.UMT5XL: None,
    Checkpoint.UMT5XXL: None,
}

dec_self_attn_out_scale_dict: dict[Checkpoint, Optional[list[float]]] = {
    # 8 layers
    Checkpoint.T5v1_1Small: None,
    # 24 layers
    Checkpoint.T5v1_1XL: None,
    # 24 layers
    Checkpoint.T5v1_1XXL: None,
    Checkpoint.MT5Small: None,
    Checkpoint.MT5Base: None,
    Checkpoint.MT5Large: None,
    Checkpoint.MT5XL: None,
    Checkpoint.MT5XXL: None,
    Checkpoint.UMT5Small: None,
    Checkpoint.UMT5Base: None,
    Checkpoint.UMT5XL: None,
    Checkpoint.UMT5XXL: None,
}

dec_cross_attn_out_scale_dict: dict[Checkpoint, Optional[list[float]]] = {
    # 8 layers
    Checkpoint.T5v1_1Small: None,
    # 24 layers
    Checkpoint.T5v1_1XL: None,
    # 24 layers
    Checkpoint.T5v1_1XXL: None,
    Checkpoint.MT5Small: None,
    Checkpoint.MT5Base: None,
    Checkpoint.MT5Large: None,
    Checkpoint.MT5XL: None,
    Checkpoint.MT5XXL: None,
    Checkpoint.UMT5Small: None,
    Checkpoint.UMT5Base: None,
    Checkpoint.UMT5XL: None,
    Checkpoint.UMT5XXL: None,
}

dec_ffn_out_scale_dict: dict[Checkpoint, Optional[list[float]]] = {
    # 8 layers
    Checkpoint.T5v1_1Small: [*[1]*2, 1/2, *[1]*5],
    # 24 layers
    Checkpoint.T5v1_1XL: None,
    # 24 layers
    Checkpoint.T5v1_1XXL: None,
    Checkpoint.MT5Small: None,
    Checkpoint.MT5Base: None,
    Checkpoint.MT5Large: None,
    Checkpoint.MT5XL: None,
    Checkpoint.MT5XXL: None,
    Checkpoint.UMT5Small: None,
    Checkpoint.UMT5Base: None,
    Checkpoint.UMT5XL: None,
    Checkpoint.UMT5XXL: None,
}

ckpt_to_hf_model_name: dict[Checkpoint, str] = {
    Checkpoint.T5v1_1Small: 'google/t5-v1_1-small',
    Checkpoint.T5v1_1XL: 'google/t5-v1_1-xl',
    Checkpoint.T5v1_1XXL: 'google/t5-v1_1-xxl',
    Checkpoint.T5v1Large: 'google-t5/t5-large',
    Checkpoint.MT5Small: 'google/mt5-small',
    Checkpoint.MT5Base: 'google/mt5-base',
    Checkpoint.MT5Large: 'google/mt5-large',
    Checkpoint.MT5XL: 'google/mt5-xl',
    Checkpoint.MT5XXL: 'google/mt5-xxl',
    Checkpoint.UMT5Small: 'google/umt5-small',
    Checkpoint.UMT5Base: 'google/umt5-base',
    Checkpoint.UMT5XL: 'google/umt5-xl',
    Checkpoint.UMT5XXL: 'google/umt5-xxl',
    Checkpoint.PileT5Large: 'EleutherAI/pile-t5-large',
}

# Model type classification
ckpt_model_type: dict[Checkpoint, str] = {
    Checkpoint.T5v1_1Small: 't5',
    Checkpoint.T5v1_1XL: 't5',
    Checkpoint.T5v1_1XXL: 't5',
    Checkpoint.T5v1Large: 't5',
    Checkpoint.MT5Small: 'mt5',
    Checkpoint.MT5Base: 'mt5',
    Checkpoint.MT5Large: 'mt5',
    Checkpoint.MT5XL: 'mt5',
    Checkpoint.MT5XXL: 'mt5',
    Checkpoint.UMT5Small: 'umt5',
    Checkpoint.UMT5Base: 'umt5',
    Checkpoint.UMT5XL: 'umt5',
    Checkpoint.UMT5XXL: 'umt5',
    Checkpoint.PileT5Large: 'umt5',  # pile-t5 is UMT5 architecture
}

# Legacy compatibility - deprecated, use ckpt_model_type instead
ckpt_is_umt5: dict[Checkpoint, bool] = {
    ckpt: (model_type == 'umt5') for ckpt, model_type in ckpt_model_type.items()
}