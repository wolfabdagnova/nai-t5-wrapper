from .t5 import (
    T5,
    T5EncoderStack,
    labels_to_decoder_input_ids,
    label_mask_to_decoder_mask,
)
from .t5_hf import hf_to_based_t5_state, hf_to_based_t5_enc_state, to_based_config
from .t5_common import T5Config
from .hf_wrapper import NAIT5EncoderModel, NAIT5EncoderOutput

__all__ = [
    # Core T5 classes
    "T5",
    "T5EncoderStack",
    "T5Config",
    # HuggingFace-compatible wrapper
    "NAIT5EncoderModel",
    "NAIT5EncoderOutput",
    # Utilities
    "labels_to_decoder_input_ids",
    "label_mask_to_decoder_mask",
    "hf_to_based_t5_state",
    "hf_to_based_t5_enc_state",
    "to_based_config",
]
